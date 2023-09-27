from collections.abc import Collection, Iterator
from collections import defaultdict as dd
from fsspec.mapping import FSMap
from functools import cached_property, partial, cache
from itertools import starmap, product
from numbers import Number, Integral as Int
from pathlib import Path

import dask.array as da
import xarray as xr
import numpy as np
import typing
import zarr
import math

from crest.src.base import BaseAbstract
from crest.src.data.loading import Block, Blockset
from crest.src.utils import Stopwatch

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

    # Ensure we include int(32/64) representations of NaN
    DEFAULT_INVALID_VALUES = [-2147483648, -9223372036854775808]



    def __init__(self,
        location      : Path | str | FSMap | xr.Dataset,
        features      : list[str]                        = [],
        extent        : dict[str, Collection]            = {},
        window_depth  : dict[str, Int | Collection[Int]] = {},
        valid_percent : dict[str | tuple[str], Number]   = {},
        invalid_value : object                           = [],
        **kwargs
    ):
        self.location = location
        self.features = features
        self.extent   = extent.copy()
        self._kwargs  = kwargs
        self._window_depth  = window_depth.copy()
        self._valid_percent = valid_percent.copy()
        self._invalid_value = invalid_value
        self._virtual_dims  = {}
        self._validate_parameters()


    def __getattr__(self, attr: str):
        """ Allow calls to be passed to the underlying xarray/dask object """
        # Prevent recursive loop when there is an AttributeError in data/dask
        if attr in ['data', 'dask']: return self.__getattribute__(attr)

        # Prevent recursive loop when pickling objects
        if attr not in ['__getstate__', '__setstate__']:
            if hasattr(self.data, attr): return getattr(self.data, attr)
            if hasattr(self.dask, attr): return getattr(self.dask, attr)
        return self.__getattribute__(attr)


    def __getstate__(self):
        return {
            'location' : self.location,
            'features' : self.features,
            'extent'   : self.extent,
            '_kwargs'  : self._kwargs,
            '_window_depth'  : self._window_depth,
            '_valid_percent' : self._valid_percent,
            '_invalid_value' : self._invalid_value,
            '_virtual_dims'  : {},
        }


    def __setstate__(self, d):
        self.__dict__.update(d)


    @cached_property
    def _raw_data(self):
        """ Only read the zarr once necessary """
        # If the given location isn't already an xr.Dataset, open it
        if isinstance(self.location, xr.Dataset):
            return self.location
        return xr.open_zarr(self.location, **self._kwargs)


    @cached_property
    def data(self) -> xr.DataArray:
        """ Loaded xarray object """
        data = self._raw_data

        # Select only requested features
        data = data[self.features or data.keys()]

        # Select only requested coordinates
        data = data.sel({k: slice(*ext) for k, ext in self.extent.items()})

        # Cast datetimes to float
        for key in data.coords:
            if np.issubdtype(data[key].dtype, np.datetime64):
                data = data.assign_coords({key: data[key].astype('float64')})

        # Cast int/uint/etc. to float in order to allow NaN values
        for key in data:
            dtype = data[key].dtype
            if np.issubdtype(dtype, np.number):
                if not np.issubdtype(dtype, np.floating):
                    min_dtype = np.promote_types(dtype, np.float16)
                    data[key] = data[key].astype(min_dtype)

        # Convert to a DataArray and ensure data is backed by dask
        data = data.to_array('features').chunk({})

        # Save the original coordinates for later return values
        self.original_dims = (list(data.coords.keys()), data.features.to_numpy())

        # Transpose dimensions so they are in the correct order, and return
        return data.transpose(*self.dims, ...)


    @cached_property
    def dims(self) -> list[str]:
        """ Ordered coordinate dimensions """
        return sorted(self._raw_data.coords.keys())


    @property
    def virtual(self) -> np.ndarray:#[bool]:
        """ Array of flags indicating virtual dimensions """
        return np.array([d in self._virtual_dims for d in self.dims])


    @cached_property
    def dtype(self) -> np.dtype:
        """ Create a composite datatype based on shapes of the data windows """
        sizes  = {'features': len(self.features or self._raw_data.keys())} 
        sizes |= self.window_total
        d_type = max(self._raw_data.dtypes.values())
        c_type = max(self._raw_data.coords.dtypes.values())
        return np.dtype(
            [('values', d_type, tuple(sizes.values()))] + 
            [('coords', np.dtype(
                [(dim, c_type, (size,)) for dim, size in sizes.items()])
        )] )


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
    def window_total(self) -> dict[str, Int]:
        """ Window total size per dimension """
        return {dim: 1+self.window_depth[dim].sum() for dim in self.dims}


    @cached_property
    def invalid_value(self) -> list[Number]:
        """ Invalid values, including any defined in the defaults """
        user_defined = np.atleast_1d(self._invalid_value).tolist()
        return user_defined + Datafile.DEFAULT_INVALID_VALUES


    @cached_property
    def is_uniform(self) -> bool:
        """ Flag indicating whether coordinates are uniform grids """
        self.resolution # Sets the is_uniform flag
        return self.is_uniform


    @cached_property
    def resolution(self) -> list:
        """ Coordinate resolution per axis.
            For example:
                [1,2,3] -> resolution of 1
                [3,6,9] -> resolution of 3

            If resolutions are non-uniform across a given axis, instead
            return the full resolution vector for each axis:
                [1,3,6,10,20] -> resolution of [2,3,4,10]

            Note that if _any_ axes are non-uniform, vectors are returned for
            _all_ axes - where those vectors will have a new final dimension
            of 2: the first being left difference, the second being right.

        """
        get_vec = lambda d: np.diff(d) if len(d) > 1 else np.zeros(1)
        vectors = [get_vec(self.data[dim]).round(5) for dim in self.dims]
        uniform = lambda vec: (vec.max()-vec.min()) < 1e-3
        setattr(self, 'is_uniform', all(map(uniform, vectors)))

        # Stack the left/right resolution for each vector 
        if not self.is_uniform:
            stack_lr = lambda vector: np.stack([
                np.r_[vector[:1], vector],
                np.r_[vector, vector[-1:]]
            ], axis=-1)
            return list(map(stack_lr, vectors))
        return [vec.min() for vec in vectors]


    @cached_property
    def max_resolution(self) -> list:
        """ Return the maximum resolution for each coordinate.
        If we have uniform grids, this will just be the resolution
        itself; otherwise it will be the maximum along each vector """
        return [np.atleast_1d(r).max() for r in self.resolution]


    @property
    def coord_array(self) -> da.Array:
        """ Coordinate meshgrid wrapped with dask """
        coords = [self.data[dim].values for dim in self.dims]
        array  = partial(da.from_array, name=False)
        coords = map(array, coords, self.dask.chunks)
        coords = da.meshgrid(*coords, indexing='ij')
        coords = da.stack(coords, axis=-1)
        coords = coords.rechunk({-1:-1})
        return coords


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
            if not self.is_uniform:
                # Using mean/median here would be better for performance,
                # but could miss some matches due to too little overlap
                res = res.min() 

            total = self.window_depth[dim].sum()
            size  = self.window_depth[dim].max()
            size += 0 if skip or (res==0) else ((max_res/2) / res)
            return (self.dims.index(dim), math.ceil(np.nan_to_num(size)))

        args = [self.dims, self.resolution, max_resolution, skip_dimension]
        if len(set(map(len, args))) > 1:
            raise ValueError(f'Args not all the same length: {args}')
        return dict(map(calculate, *args))


    def update_blocks(self, numblocks: Collection) -> list:
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
        list
            Returns the list used to rechunk the data, if it was rechunked
            (and an empty list if it wasn't rechunked).

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
        self._target_blocks = numblocks

        numblocks   = [1 if is_virtual([dim]) else n for dim, n in dim_block]
        block_shape = list(zip(numblocks, self.shape))

        # Check that all dimensions are at least as large as the requested size
        if any(block > shape for block, shape in block_shape):
            raise ValueError(f'Blocks={numblocks} > data.shape={self.shape}')

        # Calculate and apply the new chunks
        newchunks = [math.ceil(shape / block) for block, shape in block_shape]

        if self.numblocks[:-1] != tuple(numblocks):
            self.data = self.chunk(newchunks)
        else: newchunks = []

        # Ensure the new block numbers are equal to what was requested
        if any(block != db for block, db in zip(numblocks, self.numblocks)):
            raise ValueError(f'blocks={numblocks}, created {self.numblocks}')
        return newchunks


    def update_chunks(self, chunksize: Collection) -> list:
        """Rechunk data to have the requested number of chunks per dimension.

        Parameters
        ----------
        chunksize : Collection
            The chunksize for each dimension. For example, if chunksize=[3] 
            and self.data=[1, 2, 3, 4, 5] then the returned data will have 
            2 blocks: [[1,2,3], [4,5]].
            If the length of the passed chunksize collection is less than
            the total number of axes, the requested sizes are only applied
            to the first N axes (where N is the number of sizes given).

        Returns
        -------
        list
            Returns the list used to rechunk the data, if it was rechunked
            (and an empty list if it wasn't rechunked).

        Raises
        ------
        ValueError
            If a requested block count is larger than its respective dimension.
            Or, if procedure fails and the new blocks do not match the request.

        """        
        # Update virtual_dims to track the requested number of blocks
        is_virtual = lambda dim: dim[0] in self._virtual_dims
        calc_block = lambda s,c: getattr(c, '__len__', lambda: s//c)()
        dim_chunks = zip(self.dims, chunksize)
        dim_block  = zip(self.dims, map(len, map(np.atleast_1d, chunksize)))
        self._virtual_dims.update(dict(filter(is_virtual, dim_block)))
        self._target_blocks = np.array(list(map(calc_block, self.shape, chunksize)))

        chunksize   = [1 if is_virtual([dim]) else n for dim, n in dim_chunks]
        block_shape = list(zip(chunksize, self.shape))

        # Check that all dimensions are at least as large as the requested size
        # if any(block > shape for block, shape in block_shape):
        for block, shape in block_shape:
            if hasattr(block, '__len__'):
                if sum(block) != shape:
                    raise ValueError(f'Blocks={chunksize} != data.shape={self.shape}')
            else:
                if block > shape:
                    raise ValueError(f'Blocks={chunksize} != data.shape={self.shape}')

        self.data = self.chunk(chunksize)
        return chunksize


    def apply_overlap(self,
        overlaps : dict[Int, Int] | Int,
        boundary : dict[Int, Number] | Number | str = np.nan,
    ):# -> Iterator[Block]:
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
        the case in general - needs further testing. It appears the failure mode
        is when there needs to be more than one chunk included in the overlap,
        on the interior of the array:
            [[1,2], [3,4], [5,6], [7,8]] with an overlap of 3 would generate:
            [[nan, nan, nan, 1, 2, 3, 4], [1,2,3,4,5,6], [3,4,5,6,7,8], ...]
        instead of the expected:
            [[nan, nan, nan, 1, 2, 3, 4, 5], [nan, 1, 2, 3, 4, 5, 6, 7], ...]
        because the third element requires an additional chunk.
        A common case where this does not fail is when the last chunk is too
        small, as the nan filling will correctly include the required number
        of additional elements. The issue only appears when internal chunks
        are too small.

        Returns
        -------
        Blockset
            A Blockset object containing all of the Blocks for this Datafile.

        """
        # Expected number of blocks, chunk sizes, and dimension shapes
        # Skip the last dimension (features), since that varies by grid
        blocks = self._target_blocks # self.dask.numblocks[:-1]
        chunks = self.dask.chunks[:-1]
        shapes = self.dask.shape[:-1]

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

        def get_id(index):
            """ Get the block id from the multi-index representation """
            return np.ravel_multi_index(index, tuple(blocks) + (1,))

        def set_id(array):
            """ Set all values in a block equal to its own block id """
            f = lambda x, block_id: np.full_like(x, get_id(block_id), dtype='int32')
            return array.map_blocks(f, dtype='int32')

        def mask_id(array):
            """ Boolean mask indicating if a value originated in this block """
            f = lambda x, block_id: x != get_id(block_id)
            return array.map_blocks(f, dtype=bool)

        # Ensure name=False to avoid hashing all values in the array
        to_darray = partial(da.from_array, name=False)

        # Create mask indicating overlapped elements in each block
        mask = np.zeros(shapes + (1,), dtype=bool)
        mask = set_id( to_darray(mask, chunks + ((1,),)) )

        # Clip block extents to avoid duplication
        # extents = [max(c)+o*2 for c,o in zip(self.chunks, overlaps.values())]

        # Create overlapping blocks, tiling virtual dimensions where necessary
        virtual = [self._virtual_dims.get(d, 1) for d in self.dims] + [1]
        repeat  = partial(da.tile, reps=virtual)
        overlap = partial(dask_overlap, **{
            'depth'         : overlaps,
            'boundary'      : boundary,
            'allow_rechunk' : False,
        })

        # Separate data/coords/mask into independent blocks
        d_blocks = repeat( overlap(self.dask) ).blocks
        c_blocks = repeat( overlap(self.coord_array) ).blocks
        m_blocks = repeat( mask_id(overlap(mask)) ).blocks

        # Ensure we're generating the same number of blocks for all grids
        is_equal = lambda a: np.prod(a) == np.prod(blocks)
        assert(is_equal(d_blocks.shape[:-1])), [d_blocks.shape, blocks]
        assert(is_equal(c_blocks.shape[:-1])), [c_blocks.shape, blocks]
        assert(is_equal(m_blocks.shape[:-1])), [m_blocks.shape, blocks]        

        # Define args/kwargs for the Block objects that will be created
        block_grids = [d_blocks, c_blocks, m_blocks]
        block_kwarg = {
            'dims'          : self.dims,
            'original_dims' : self.original_dims,
            'window_depth'  : self.window_depth,
            'valid_percent' : self.valid_percent,
            'invalid_value' : self.invalid_value,
        }

        # Non-uniform coordinate grids use different resolutions in each block
        if not self.is_uniform:
            get_value = lambda i, d: d[i] if isinstance(d, dict) else d
            to_blocks = lambda i, a: da.tile( dask_overlap(a, **{
                'depth'         : {0: get_value(i, overlaps)}, 
                'boundary'      : {0: get_value(i, boundary)},
                'allow_rechunk' : False,
            }), reps=[virtual[i], 1] ).blocks

            # Create a BlockView object for each coordinate vector
            r_chunks = [[chunk, (2,)] for chunk in chunks] # 2 = left/right
            r_arrays = map(to_darray, self.resolution, r_chunks)
            r_blocks = list(starmap(to_blocks, enumerate(r_arrays)))
            r_counts = [r.size for r in r_blocks]
            assert(is_equal(r_counts)), [r_counts, blocks]
            
            # Wrap the vectors to be indexable as a cartesian product array
            fetch = lambda _, idx: tuple(r[i] for r,i in zip(r_blocks, idx))
            Array = type('product_array', (object,), {'__getitem__': fetch})
            block_grids.append(Array())

        # Uniform resolution coordinates use the same resolution across blocks
        else: block_kwarg['resolution'] = self.resolution 

        # Returns list of functions to allow lazy creation of the Block objects
        gen_block = lambda i: Block(*[g[i] for g in block_grids], **block_kwarg)
        lazy_func = lambda i: cache(lambda: gen_block(i))
        return map(lazy_func, product(*map(range, blocks)))


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

        # elif isinstance(self.location, FSMap):
        #     if not self.location.fs.exists(self.location.root):
        #         raise FileNotFoundError(f'Database not found: {self.location}')

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
