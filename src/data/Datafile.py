from collections.abc import Collection
from collections import defaultdict as dd
from fsspec.mapping import FSMap 
from functools import cached_property
from itertools import starmap
from numbers import Number, Integral as Int
from pathlib import Path 

import dask.array as da 
import xarray as xr
import numpy as np
import typing
import zarr
import math 

from crest.src.base import BaseAbstract 
from crest.src.data import Block, Blockset

# Bool type which allows numpy bools as well
Bool = bool | np.bool_


class Datafile(BaseAbstract):
    """Class which handles loading data from a single source.
    
    Parameters
    ----------
    location      : Path | str | FSMap | xr.Dataset,
        Location for which to load the data from. Note that this can take
        a variety of formats, including disk filepath, or S3 bucket via 
        an FSMap object. As well, an already loaded xr.Dataset object can 
        also be passed in.
    features      : list[str]
        The list of features (variables) to load / keep in the created 
        xr and dask objects.
    extent        : dict[str, Collection[Number]]
        A dictionary mapping {Dimension: [lower bound, upper bound]} for
        loading the data. For example, {'latitude': [10, 20]} would indicate
        only data from 10 degrees latitude to 20 degrees latitude should be 
        loaded.
    window_depth  : dict[str, Int | Collection[Int]]
        Format of {Dimension: (lower, upper)} or {Dimension: both}, where 
        (lower, upper) defines the number of elements to the left and to the 
        right of the window center, and both would indicate both=left=right. 
        {'time': (1, 0)} would indicate a window which has two elements along
        the time dimension, where one element is the center and the other is 
        a single lookback step (to the left). {'time': 1} would be equivalent
        to {'time': (1,1)}. Dimensions that exist in the data but are not given
        in this dictionary use Datafile.DEFAULT_WINDOW_SIZE by default. 
    valid_percent : dict[str | tuple[str], Number] 
        Format of {Dimension: percent} or {(Dim1, Dim2, ...): percent}, where 
        the key can be single string, or a tuple with one or more elements;
        multiple dimensions in the key tuple indicate that the valid percent 
        is applied across all of those dimensions combined.
        Percent is a value in the range [0, 1], indicating the percentage of 
        a window (for the given dimension(s)) which needs to be valid in order
        for the window itself to be valid. For example, 
        {('latitude', 'longitude'): 0.8, ('time',): 1} would indicate that for
        a given window, at least 80% of the elements must be valid across the 
        2d grid of latitude x longitude, and subsequently all elements across
        the time dimension must be valid. Note that the dictionary is ordered,
        and so dimensions are evaluated in the order given.
    invalid_value : Number | Collection[Number]
        A single value or a collection of values which should be treated as 
        NaN in the `data`. Note that np.NaN and its integer representation
        (i.e. -2147483648) are always treated as NaN, regardless of this 
        parameter. 
    **kwargs
        Any additional kwargs are passed into xr.open_zarr when loading the 
        given `location` (assuming it is not already an xr.Dataset object).

    Raises
    ------
    FileNotFoundError
        If the requested str/Path `location` does not exist.
    ValueError
        - If a given extent does not have two elements
        - If a given window depth has more than two elements
        - If a given valid percent is not in the range [0,1]

    """
    # Window size is a single element for undefined dimensions
    DEFAULT_WINDOW_SIZE = 0

    # Full window must be valid for undefined dimensions
    DEFAULT_VALID_PERCENT = 1 

    # Ensure we include integer representation of NaN (-2147483648)
    DEFAULT_INVALID_VALUES = [-2147483648]



    def __init__(self,
        location      : Path | str | FSMap | xr.Dataset,
        features      : list[str]                        = [],
        extent        : dict[str, Collection[Number]]    = {},
        window_depth  : dict[str, Int | Collection[Int]] = {},
        valid_percent : dict[str | tuple[str], Number]   = {},
        invalid_value : Number | Collection[Number]      = [],
        **kwargs
    ):
        self.location = location
        self.features = features
        self.extent   = extent.copy()
        self._window_depth  = window_depth.copy()
        self._valid_percent = valid_percent.copy()
        self._invalid_value = invalid_value
        self._raw_data      = self.location
        self._virtual_dims  = {}
        self._validate_parameters()

        # If the given location isn't already an xr.Dataset, open it
        if not isinstance(self.location, xr.Dataset): 
            self._raw_data = xr.open_zarr(self.location, **kwargs)
    
    
    def __getattr__(self, attr: str):
        """ Allow calls to be passed to the underlying xarray/dask object """
        if attr not in ['__getstate__', '__setstate__']:
            if hasattr(self.data, attr): return getattr(self.data, attr)
            if hasattr(self.dask, attr): return getattr(self.dask, attr)
        raise AttributeError(f"'Datafile' object has no attribute '{attr}'")


    @cached_property
    def data(self) -> xr.DataArray:
        """ Loaded xarray object """
        data = self._raw_data

        # Select only relevant features
        data = data[self.features or data.keys()]

        # Select only relevant coordinates
        data = data.sel({k: slice(*ext) for k, ext in self.extent.items()})
        
        # Convert to a DataArray and rechunk along the new axis
        data = data.to_array('features').chunk({'features': -1})

        # Save the original coordinates for later return values
        self.original_dims = (data.coords.keys(), data.features)

        # Transpose dimensions so they are in the correct order, and return
        return data.transpose(*self.dims, ...)


    @cached_property
    def dims(self) -> list[str]:
        """ Ordered coordinate dimensions """
        return sorted(self._raw_data.coords.keys())
    

    @cached_property
    def valid_percent(self) -> dict[str, Number]:
        """ {Dimension key : valid percent} with default value for missing.
            Keys are also expanded into tuples, and missing dims are added. """
        default = Datafile.DEFAULT_VALID_PERCENT
        expand  = lambda k: tuple(np.atleast_1d([k]).flatten())
        mapping = {expand(dim): v for dim, v in self._valid_percent.items()}
        defined = np.hstack(mapping | {'':''})
        mapping|= {(dim,): default for dim in self.dims if dim not in defined}
        return dd(lambda: default, mapping)

        
    @cached_property
    def window_depth(self) -> dict[str, Int]:
        """ {Dimension key : window depth} with default value for missing.
            Values are also expanded into (left side, right side) formats. """
        flatten = lambda v: np.array([v]).flatten().astype(int)
        expand  = lambda v: np.array([flatten(v)[0], flatten(v)[-1]])
        mapping = {dim: expand(v) for dim, v in self._window_depth.items()}
        return dd(lambda: expand(Datafile.DEFAULT_WINDOW_SIZE), mapping)


    @cached_property
    def invalid_value(self) -> list[Number]:
        """ Invalid values, including any defined in the defaults """
        user_defined = np.atleast_1d(self._invalid_value).tolist()
        return user_defined + Datafile.DEFAULT_INVALID_VALUES


    @cached_property
    def resolution(self) -> list[Number]:
        """ Coordinate resolution per axis.
            For example: 
                [1,2,3] -> resolution of 1
                [3,6,9] -> resolution of 3
        """
        get_resolution = lambda v: (v.max()-v.min()).values/((len(v)-1.) or 1)
        return [get_resolution(self.data[a]).round(6) for a in self.dims]


    @property
    def coord_array(self) -> da.Array:
        """ Coordinate meshgrid wrapped with dask """
        coords = [self.data[dim].values for dim in self.dims]
        coords = np.meshgrid(*coords, indexing='ij')
        coords = np.stack(coords, axis=-1)
        chunks = self.chunksize[:-1] + (-1,)
        return xr.DataArray(coords).chunk(chunks).data


    @property
    def dask(self) -> da.Array:
        """ DaskArray reference held by the xr.DataArray """
        return self.data.data
    

    def ensure_dims(self, dims: set[str]) -> None:
        """ Ensure any missing dimensions are added as virtual dimensions """
        missing = dims - set(self.dims)

        if missing:
            # Add the new dimension(s) to the raw data and valid_percents
            missing_to_nan = dict(zip(missing, [[np.nan]]*len(missing)))
            self._raw_data = self._raw_data.expand_dims(dim=missing_to_nan)

            # Add the new dimension(s) to virtual_dims to track numblocks
            self._virtual_dims.update(missing_to_nan)
            self.dims = sorted(dims)

            # Force a cache refresh for these values
            for key in ['data', 'valid_percent']:
                self.__dict__.pop(key, None)


    def calculate_overlap(self,
        max_resolution : Collection, 
        skip_dimension : Collection[Bool],
    ) -> dict[Int, Int]:
        """Calculate the required block overlap for the given max resolutions.

        Parameters
        ----------
        max_resolution : Collection
            Max resolution per dimension across all Datafiles in a Dataset.
        skip_dimension : Collection[bool]
            Bool flag per dimension which indicates if we should just return
            the originally requested window size as the block overlap buffer.
            This is used to ensure exactly one coordinate grid maintains the
            minimum necessary buffer for the Dataset. If this wasn't present, 
            there would be many duplicate samples created from the additional 
            overlap (e.g. if there were multiple grids with equal resolution,
            they all would have window_depth+1 as their overlap and thus have
            duplicate samples returned along block boundaries). With exactly 
            one grid being skipped along each dimension, it ensures that 
            duplicate samples are not created along the block edges.

        Returns
        -------
        dict
            A dictionary mapping {axis number: overlap depth}.

        Raises
        ------
        ValueError
            If max_resolution, skip_dimension, self.dims, and self.resolution
            aren't all the same length.

        """
        def calculate(dim, res, max_res, skip):
            total = self.window_depth[dim].sum()
            size  = self.window_depth[dim].max() 
            size += 0 if skip or (res==0) else ((max_res/2) / res)
            return (self.dims.index(dim), math.ceil(np.nan_to_num(size)))

        args = [self.dims, self.resolution, max_resolution, skip_dimension]
        if len(set(map(len, args))) > 1: 
            raise ValueError(f'Args not all the same length: {args}') 
        return dict(map(calculate, *args))


    def update_chunks(self, numblocks: Collection[Int]) -> 'Datafile':
        """Rechunk data to have the requested number of blocks per dimension.

        Parameters
        ----------
        numblocks : Collection[int]
            The number of blocks which should be in each dimension. For
            example, if numblocks=[3] and self.data=[1, 2, 3, 4, 5] then the 
            returned data will have a chunksize of 2: [[1,2], [3,4], [5]].
            If the length of the passed block size collection is less than
            the total number of axes, the requested sizes are only applied
            to the first N axes (where N is the number of sizes given).

        Returns
        -------
        Datafile
            Returns the current Datafile object (self). 

        Raises
        ------
        ValueError
            If a requested block count is larger than its respective dimension.
            Or, if procedure fails and the new blocks do not match the request.

        """
        # Update virtual_dims to track the requested number of blocks
        is_virtual = lambda dim: dim[0] in self._virtual_dims
        dim_block  = list(zip(self.dims, numblocks))
        self._virtual_dims.update(dict(filter(is_virtual, dim_block)))

        numblocks   = [1 if is_virtual([dim]) else n for dim, n in dim_block]
        block_shape = list(zip(numblocks, self.shape))

        # Check that all dimensions are at least as large as the requested size
        if any(block > shape for block, shape in block_shape):
            raise ValueError(f'Blocks={numblocks} > data.shape={self.shape}')

        # Calculate and apply the new chunks
        newchunks = [math.ceil(shape / block) for block, shape in block_shape]
        self.data = self.chunk(newchunks)

        # Ensure the new block numbers are equal to what was requested
        if any(block != db for block, db in zip(numblocks, self.numblocks)):
            raise ValueError(f'blocks={numblocks}, created {self.numblocks}')
        return self 


    def apply_overlap(self, 
        overlaps : dict[Int, Int] | Int,
        boundary : dict[Int, Number] | Number | str = np.nan,
    ) -> Blockset:
        """Create a Blockset containing Blocks with the overlap applied.
        
        Parameters
        ----------
        overlaps  : dict[Int, Int] | Int
            {axis number: overlap size} dictionary which defines the amount
            of overlapping elements (on each side) for all blocks in the 
            dask array. Passing a single value will be applied to all axes.
        boundary : dict[Int, Number] | Number | str
            {axis number: boundary condition} dictionary which defines the
            boundary condition for each block axis. Passing a single value 
            will be applied to all axes. Options are 'reflect', 'periodic',
            'nearest', 'none', or an array value (so that value will fill
            the boundary). For more information, see the dask documentation:
            https://docs.dask.org/en/stable/generated/dask.array.overlap.overlap.html
        
        Notes
        -----
        The total number of blocks must remain constant, which means rechunking
        cannot be performed by the overlap operation. Current tests seem to 
        indicate bypassing the dask.overlap.overlap restriction on minimum
        chunk size will still generate correct results, but this may not be 
        the case in general - needs further testing.

        Returns
        -------
        Blockset
            A Blockset object containing all of the Blocks for this Datafile. 

        """
        def dask_overlap(x, depth, boundary, *, allow_rechunk=False):
            """ Rewrite dask.overlap.overlap to remove forced rechunking """
            from dask.array.overlap import coerce_depth, coerce_boundary
            from dask.array.overlap import boundaries, overlap_internal
            depth = coerce_depth(x.ndim, depth)
            bound = coerce_boundary(x.ndim, boundary)
            x2 = boundaries(x, depth, bound)
            x3 = overlap_internal(x2, depth)
            over = lambda k: int(bound.get(k, "none") != "none")
            trim = {k: v*2*over(k) for k,v in depth.items()}
            return da.chunk.trim(x3, trim)

        kwargs = {
            'depth'         : overlaps, 
            'boundary'      : boundary, 
            'allow_rechunk' : False,
        }

        # Tile on dimensions which are virtual 
        repeats = [self._virtual_dims.get(d, 1) for d in self.dims]

        # Clip block extents to avoid duplication
        extents = [min(c)+o*2 for c,o in zip(self.chunks, overlaps.values())]

        # Create overlapping blocks, tiling and clipping dims as necessary
        overlap = lambda a: dask_overlap(a, **kwargs)
        tile    = lambda a: da.tile(a, repeats+[1]).blocks.ravel()
        clip    = lambda a: a[tuple(map(slice, extents))]
        create  = lambda a: list(map(clip, tile( overlap(a) )))

        d_blocks = create(self.dask)
        c_blocks = create(self.coord_array)
        assert(len(d_blocks) == len(c_blocks)), [len(d_blocks), len(c_blocks)]

        kwargs = {
            'dims'          : self.dims,
            'original_dims' : self.original_dims,
            'resolution'    : self.resolution,
            'window_depth'  : self.window_depth,
            'valid_percent' : self.valid_percent,
            'invalid_value' : self.invalid_value, 
        }
        gen_blocks = lambda data, coords: Block(data, coords, **kwargs)
        block_objs = map(gen_blocks, d_blocks, c_blocks)
        return Blockset(list(block_objs))


    def _validate_parameters(self) -> None:
        """Validate parameters given as input to the Datafile object.

        Raises
        ------
        FileNotFoundError
            If the requested str/Path `location` does not exist.
        ValueError
            - If a given extent does not have two elements
            - If a given window depth has more than two elements
            - If a given valid percent is not in the range [0,1]

        """
        if not isinstance(self.location, (xr.Dataset, FSMap)):
            if not Path(self.location).exists():
                raise FileNotFoundError(f'File not found: {self.location}')
            self.location = zarr.DirectoryStore(self.location)

        for key, extent in self.extent.items():
            if len(extent) != 2:
                message = f'{key} extent must be [start, end]: {extent}'
                raise ValueError(message)

        for key, window in self._window_depth.items():
            if (not isinstance(window, Int)) and (len(window) > 2):
                message = f'{key} window must be (before, after): {window}'
                raise ValueError(message)

        for key, percent in self._valid_percent.items():
            if not (0 <= percent <= 1):
                message = f'{key} percent must be in range [0, 1]: {percent}'
                raise ValueError(message)
