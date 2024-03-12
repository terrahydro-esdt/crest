from dask.diagnostics import ProgressBar
from collections.abc import Collection, Iterator, Callable
from collections import defaultdict as dd
from fsspec.mapping import FSMap
from functools import cached_property, partial, cache
from itertools import starmap, product
from numbers import Number, Integral as Int
from pathlib import Path
from logging import Logger
from typing import Union 

import dask.array as da
import xarray as xr
import pandas as pd 
import numpy as np
import warnings
import hashlib
import typing
import shutil
import zarr
import math
import dask 

from crest.base import BaseAbstract
from .Block import Block
from .Blockset import Blockset

# Bool type which allows numpy bools as well
Bool = Union[bool, np.bool_]


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
    preprocessors : list[Callable]
        List of functions that are applied to the data (lazily) upon loading.
        For example, a function could be given which filters the data based
        on the value of a feature. The function signature should be::

            def function(datafile: Datafile, data: xr.Dataset) -> xr.Dataset
        
        where the current datafile object will be given as the first parameter
        (to allow interacting with features, extent, etc.); and the data itself
        as the second parameter (as the opened xr.Dataset object). Any 
        modifications should be applied to the data object inside the function,
        and that data object then used as the return value of the function.
        Example::

            def filter_soilmoisture(datafile, data: xr.Dataset):
                return data.where(data['soil_moisture'] >= 0.02)

        Note that any functions given must be picklable if multiprocessing is
        used with the CREST Batcher class (thus lambda functions are invalid).
    sort_dims     : bool
        Whether samples generated from this Datafile should sort dimensions so
        that dimension ordering is consistent across all Datafiles (default);
        or if the original dimension order should be maintained in any samples 
        that are generated (which might lead to different orderings across
        Datafiles). 
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
        location      : Union[Path, str, FSMap, xr.Dataset],
        features      : list[str]                        = [],
        extent        : dict[str, Collection]            = {},
        window_depth  : dict[str, Int | Collection[Int]] = {},
        valid_percent : dict[str | tuple[str], Number]   = {},
        invalid_value : object                           = [],
        preprocessors : list[Callable]                   = [],
        sort_dims     : bool                             = True,
        **kwargs
    ):
        self.location = location
        self.features = sorted(features)
        self.extent   = extent.copy()
        self._kwargs  = kwargs
        self._window_depth  = window_depth.copy()
        self._valid_percent = valid_percent.copy()
        self._invalid_value = invalid_value
        self.preprocessors  = preprocessors
        self.sort_dims      = sort_dims
        self.dataset_index  = 0 

        # Store initialization parameter names for pickling
        self._init_keys = list(self.__dict__) + ['_init_keys']

        # Verify all given parameters are valid
        self._validate_parameters()


    def __getattr__(self, attr: str):
        """ Allow calls to be passed to the underlying xarray/dask object """
        if hasattr(Datafile, attr): return self.__getattribute__(attr)

        # Prevent recursive loop when pickling objects
        if attr not in ['data', 'dask', '__getstate__', '__setstate__']:
            try:
                if hasattr(self.data, attr): return getattr(self.data, attr)
                if hasattr(self.dask, attr): return getattr(self.dask, attr)
            except AttributeError: return self.__getattribute__(attr)
            except Exception as e1: 
                try: return self.__getattribute__(attr)
                except Exception as e2: raise e1 from e2 
        return self.__getattribute__(attr)


    def __getstate__(self):
        """ Ensure unpickle-able objects are not included """
        return {key: getattr(self, key) for key in self._init_keys}


    def __repr__(self) -> str:
        return f'Datafile[{self.label}]'


    @cached_property
    def _raw_data(self):
        """ Only read the zarr once necessary """
        if isinstance(self.location, xr.Dataset):
            raw = self.location
        else:
            # If the given location isn't already an xr.Dataset, open it
            if not isinstance(self.location, FSMap):
                location = zarr.DirectoryStore(self.location)
            else: location = self.location
            raw = xr.open_zarr(location, **self._kwargs)

        # Handle the summary statistics 
        if 'summary' in raw:

            # If no extents are changed / preprocessors applied, raw_data statistics
            # will be the same as data statistics
            if not (len(self.preprocessors) or len(self.extent)):
                self.__dict__['summary'] = raw['summary']
            raw = raw.drop_vars(['summary', 'features', 'statistics'], errors='ignore')

            # Set datetime dtype back to np.datetime64 if it was previously converted
            if hasattr(raw, 'datetime') and np.issubdtype(raw.datetime.dtype, np.float32):
                raw = raw.assign_coords(datetime=raw.datetime.astype('datetime64[m]'))
        return raw 


    @cached_property
    def data(self) -> xr.DataArray:
        """ Loaded xarray object """
        data = self._raw_data

        # Apply any preprocessing functions
        for func in self.preprocessors:
            data = func(self, data)

        # Select only requested features
        data = data[self.features or sorted(data.keys())]

        # Remove summary statistics
        data = data.drop_vars(['summary', 'features', 'statistics'], errors='ignore')

        # Select only requested coordinates
        data = data.sel({k: slice(*ext) for k, ext in self.extent.items() 
                        if k in data and k not in self._virtual_dims})

        # Store original data dtypes - all coords are converted to float in 
        # Datafile._typed_data. In order to allow datetime64 to fit float32,
        # we convert datetime values to datetime64[m] (i.e. minute resolution)
        dtypes = {k: T if not np.issubdtype(T := data[k].dtype, np.datetime64)
                    else 'datetime64[m]' for k in list(data)+list(data.coords)}

        # Convert to a DataArray and ensure data is backed by dask
        data = data.to_array('features').chunk({})

        # Add Datafile configuration to the attributes
        if 'Datafile.config' not in data.attrs:
            data.attrs.update({
                'Datafile.config'      : self.config,
                'Datafile.config_hash' : self.config_hash,
            })

        # Save the original coordinates/dtypes for later return values
        original_dims = [c for c in data.coords if c not in self._virtual_dims]
        self.original_dims = (
            sorted(original_dims) if self.sort_dims else original_dims, 
            data.features.to_numpy(),
            dtypes,
            self.features,
        )

        # Sanity check - Sample.py assumes features are sorted
        assert(sorted(self.original_dims[1]) == list(self.original_dims[1]))

        # Transpose dimensions so they are in the correct order, and return
        order = sorted(set(data.coords) - {'features'})
        return data.transpose(*order, ...)


    @cached_property
    def summary(self) -> xr.DataArray:
        """ Return summary statistics for data """
        # Ensure cached summary data is used if available
        self._raw_data
        if 'summary' in self.__dict__:
            return self.__dict__['summary']
        data = self.data.to_dataset('features')

        # Include in the summary stats any coordinates requested as features
        for coord in data.coords:
            if (coord in self.features) and (coord != 'datetime'):
                coords = {c: data[c] for c in data.coords if c != coord}
                data[f'{coord}_f'] = data[coord]
                data[f'{coord}_f'] = data[[f'{coord}_f']] \
                    .expand_dims(**coords)[f'{coord}_f'] \
                    .transpose(*list(data.dims)) \
                    .chunk(self.data.data.chunksize[:-1])

        extra = dd(int, {'null' : data.isnull().sum})
        stats = ['mean', 'std', 'min', 'max'] + list(extra)
        coord = xr.Variable('statistics', stats) 
        value = lambda k: getattr(data, k, extra[k])().to_array('features')
        stats = xr.concat(map(value, stats), coord).to_dataset('statistics')

        # Compute percentiles as a group for efficiency
        stats[['p25', 'median', 'p75']] = xr.apply_ufunc(
            lambda x: da.percentile(x.ravel(), [25, 50, 75], internal_method='tdigest'),
            data, **{
                'dask'             : 'allowed',
                'input_core_dims'  : [list(data.dims)],
                'output_core_dims' : [['statistics']],
            }).to_array('features').to_dataset('statistics')
        stats = stats.to_array('statistics').to_dataset('features')
        stats = stats.rename({f'{c}_f':c for c in data.coords if f'{c}_f' in stats})
        return stats.to_array('features')


    @property
    def _typed_data(self) -> xr.DataArray:
        """ Data/coords converted to float types """
        data = self.data

        # Cast int/uint/etc. to float in order to allow NaNs
        # for key in data.features:
        if np.issubdtype(data.dtype, np.number):
            if not np.issubdtype(data.dtype, np.floating):
                data = data.astype( np.promote_types(data.dtype, np.float16) )

        # Coordinates must have uniform type, so we cast to float32
        # Note: datetime64 are converted to datetime64[m] (minute resolution)
        for key in data.coords:
            if key != 'features':
                if np.issubdtype(data[key].dtype, np.datetime64):
                    data = data.assign_coords({key: data[key].astype('float64')/6e10})
                data = data.assign_coords({key: data[key].astype('float32')})
        return data


    @property
    def dask(self) -> da.Array:
        """ DaskArray reference held by the xr.DataArray """
        return self._typed_data.data
    

    @property
    def dask_coords(self) -> da.Array:
        """ Coordinate meshgrid wrapped with dask """
        todask = partial(da.from_array, name=False)
        coords = (self._typed_data[d].values for d in self.dims)
        c_dask = map(todask, coords, self.dask.chunks)
        c_grid = da.meshgrid(*c_dask, indexing='ij')
        return da.stack(c_grid, axis=-1).rechunk({-1:-1})


    @property
    def name(self) -> str:
        """ Return a name for this Datafile using the location if possible """
        if isinstance(self.location, (Path, str)):
            loc = Path(self.location)
            if 'Cache' in loc.parts and len(loc.stem) == len(self.config_hash):
                return f'{loc.parent.stem}_{loc.stem[:4]}'
            return loc.stem
        return f'{type(self.location).__name__}_{self.config_hash[:4]}'


    @property
    def label(self) -> str:
        """ Label is the combination of Dataset index and Datafile name """
        return f'{self.dataset_index}:{self.name}'


    @property
    def config(self) -> dict[str, str]:
        """ JSON serializable representation of the Datafile configuration """
        def f_repr(f):
            """ Get the name of the function, and a hash of its bytecode """
            name = getattr(f, '__name__', str(f))
            try: code = hashlib.sha256(f.__code__.co_code).hexdigest()
            except Exception as code: pass
            return f'{name}: {code}'

        config = dict(self.__getstate__())
        config['preprocessors'] = list(map(f_repr, config['preprocessors']))
        return {k: str(v) for k,v in config.items()}


    @property
    def config_hash(self) -> str:
        """ Hash of the config dictionary, to use as a condensed label """
        return hashlib.sha256(str(self.config).encode('utf-8')).hexdigest()


    @cached_property
    def dims(self) -> list[str]:
        """ Ordered coordinate dimensions """
        return sorted(set(self.data.coords) - {'features'})


    @cached_property
    def _virtual_dims(self) -> dict:
        """ Maps {Dimension key : number of blocks} for virtual dimensions.
            A dimension is virtual when it does not exist in this Datafile,
            but does exist in other Datafiles that this Datafile is combined
            with in a Dataset. """
        return dict()


    @property
    def virtual(self) -> np.ndarray:#[bool]:
        """ Array of flags indicating virtual dimensions """
        return np.array([d in self._virtual_dims for d in self.dims])


    @cached_property
    def dtype(self) -> np.dtype:
        """ Create a composite datatype based on shapes of the data windows """
        sizes = {'features': len(self.data.features)} 
        sizes|= self.window_total
        ctype = lambda dim: (dim, self.data[dim].dtype, (sizes[dim],))
        return np.dtype(
            [('values', self.data.dtype, tuple(sizes.values()))] +
            [('coords', np.dtype(list(map(ctype, sizes))))]
        )


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
            
            For example::

                [1,2,3] -> resolution of 1
                [3,6,9] -> resolution of 3

            If resolutions are non-uniform across a given axis, instead
            return the full resolution vector for each axis::
            
                [1,3,6,10,20] -> resolution of [2,3,4,10]

            Note that if _any_ axes are non-uniform, vectors are returned for
            _all_ axes - where those vectors will have a new final dimension
            of 2: the first being left difference, the second being right.

        """
        get_vec = lambda d: np.diff(d) if len(d) > 1 else np.zeros(1)
        vectors = [get_vec(self._typed_data[dim]).round(5) for dim in self.dims]
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
        dimension_blks : Collection | None = None,
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
        def calculate(dim, res, max_res, skip, blocks):
            if not self.is_uniform:
                # Using mean/median here would be better for performance,
                # but could miss some matches due to too little overlap
                res = res.min() 

            if blocks > 1:
                total = self.window_depth[dim].sum()
                size  = self.window_depth[dim].max()
                size += 0 if skip or (res==0) else ((max_res/2) / res)
            else: size = 0
            return (self.dims.index(dim), math.ceil(np.nan_to_num(size)))

        blks = dimension_blks if dimension_blks is not None else [2] * len(self.dims)
        args = [self.dims, self.resolution, max_resolution, skip_dimension, blks]
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
        # Add 1 for any unspecified dims (except features)
        numblocks = list(numblocks) + [1] * (len(self.dask.numblocks) - (len(numblocks)+1))

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
        # Add -1 for any unspecified dims (except features)
        chunksize = list(chunksize) + [-1] * (len(self.dask.chunks) - (len(chunksize)+1))

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
        optimize : bool = True,
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
        optimize : bool
            Whether dask.optimize should be used on the full task graph. This
            can speed up access when batching data, but incurs a higher cost
            when initially generating sample blocks.

        Notes
        -----
        The total number of blocks must remain constant, which means rechunking
        cannot be performed by the overlap operation. Current tests seem to
        indicate bypassing the dask.overlap.overlap restriction on minimum
        chunk size will still generate correct results, but this may not be
        the case in general - needs further testing. It appears the failure mode
        is when there needs to be more than one chunk included in the overlap,
        on the interior of the array::

            [[1,2], [3,4], [5,6], [7,8]] with an overlap of 3 would generate:
            [[nan, nan, nan, 1, 2, 3, 4], [1,2,3,4,5,6], [3,4,5,6,7,8], ...]
        
        instead of the expected::
        
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
            f = lambda x, block_id: np.full_like(x, get_id(block_id), 'int32')
            return array.map_blocks(f, dtype='int32')

        def mask_id(array):
            """ Boolean mask indicating if a value originated in this block """
            f = lambda x, block_id: x != get_id(block_id)
            return array.map_blocks(f, dtype=bool)

        # Ensure name=False to avoid hashing all values in the array
        to_darray = partial(da.from_array, name=False)
        
        # Create mask indicating out-of-bound elements in each block
        inbound_mask = np.ones(shapes + (1,), dtype=bool)
        inbound_mask = to_darray(inbound_mask, chunks + ((1,),))

        # Create mask indicating overlapped elements in each block
        overlap_mask = np.zeros(shapes + (1,), dtype=bool)
        overlap_mask = set_id( to_darray(overlap_mask, chunks + ((1,),)) )

        # Clip block extents to avoid duplication
        # extents = [max(c)+o*2 for c,o in zip(self.chunks, overlaps.values())]

        # Create overlapping blocks, tiling virtual dimensions where necessary
        virtual = [self._virtual_dims.get(d, 1) for d in self.dims] + [1]
        repeat  = partial(da.tile, reps=virtual)
        daskopt = lambda x: dask.optimize(x)[0] if optimize else x
        blocker = lambda x: daskopt(repeat(x)).blocks
        overlap = partial(dask_overlap, **{
            'depth'         : overlaps,
            'boundary'      : boundary,
            'allow_rechunk' : False,
        })

        # Separate data/coords/masks into independent blocks
        d_blocks = blocker( overlap(self.dask) )
        c_blocks = blocker( overlap(self.dask_coords) )
        i_blocks = blocker( overlap(inbound_mask, boundary=0) )
        o_blocks = blocker( mask_id(overlap(overlap_mask, boundary=-1)) )

        # Define args/kwargs for the Block objects that will be created
        block_grids = [d_blocks, c_blocks, i_blocks, o_blocks]
        block_kwarg = {
            'dims'          : self.dims,
            'original_dims' : self.original_dims,
            'window_depth'  : self.window_depth,
            'valid_percent' : self.valid_percent,
            'invalid_value' : self.invalid_value,
            'label'         : self.label,
        }

        # Ensure we're generating the same number of blocks for all grids
        is_equal = lambda a: np.prod(a) == np.prod(blocks)
        b_shapes = [block.shape[:-1] for block in block_grids]
        assert(all(map(is_equal, b_shapes))), [blocks, b_shapes]     

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
        gen_block = lambda i: (lambda: Block(*[g[i] for g in block_grids], **block_kwarg))
        lazy_func = gen_block#lambda i: cache(lambda: gen_block(i))
        return map(lazy_func, product(*map(range, blocks)))


    def reset(self, **kwargs): 
        """ Reset Datafile state to remove cached properties.
        
        Parameters
        ----------
        **kwargs
            Any __init__ parameters that should be updated in the config state.

        """
        config = self.__getstate__() | kwargs
        self.__dict__.clear()
        self.__dict__.update(config)


    def _cache(self, 
        overwrite : bool = False, 
        cache_dir : Path | str  = 'Cache',
    ):
        """Cache data in a new zarr database for faster access.

        Parameters
        ----------
        overwrite : bool
            Flag indicating whether already cached data should be overwritten.
            If False, metadata (but not the data values themselves) are checked
            against the cached data to verify it is the same. This does not
            verify that the data itself is the same, and so changing e.g. one
            of the preprocessor function definitions, might result in the wrong 
            data being used. 
        cache_dir : Path | str
            Location for the cached data to be stored. By default, data is 
            cached in `./Cache`. 

        """
        data = self.data.to_dataset('features')
        dest = Path(cache_dir, self.name, f'{self.config_hash}.zarr')
        dest.parent.mkdir(exist_ok=True, parents=True)

        # Erase virtual dimensions
        for dim in data.coords:
            if dim in [d for d,v in zip(self.dims, self.virtual) if v]:
                data = data.isel({dim: 0}, drop=True)

        # Include summary statistics
        data['summary'] = self.summary

        # Verify cached data is equivalent
        if (not overwrite) and dest.exists():
            try: 
                with xr.open_zarr(dest) as cache:
                    overwrite = not all(
                        getattr(data, attr) == getattr(cache, attr)
                        for attr in ['dims', 'attrs', 'nbytes', 'chunksizes']
                    ) and bool(xr.align(data, cache, join='exact'))
            except: overwrite = True

        # Reinitialize this Datafile with the new cache
        # Anything handled by the cache (e.g. extent) can be dropped
        self.reset(**{
            'location'      : dest,
            # 'features'      : [],
            'extent'        : {},
            'preprocessors' : [],
        })

        # Write the data to the destination
        if overwrite or (not dest.exists()):
            if dest.exists(): shutil.rmtree(dest)
            print(f'\nCaching {self.name}...')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with ProgressBar(): data.to_zarr(dest, encoding={})
        else: print(f'Cache exists for {self.name}')


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
