from collections.abc import Collection, Callable, Sequence
from functools import cached_property, reduce
from itertools import starmap
from numbers import Number, Integral as Int
from logging import Logger 
from typing import Union 

import dask.array as da 
import bottleneck as bn
import xarray as xr
import pandas as pd
import numpy as np 

from crest.base import BaseAbstract
from crest.utils import Stopwatch


class Block(BaseAbstract):
    """Class which wraps a dask block. 

    Parameters
    ----------
    data          : da.Array
        Dask Array object containing the data within the block.
    coords        : da.Array
        Dask Array object containing the coordinates which define the block.
    inbound_mask  : da.Array
        Dask Array object indicating which elements are inside the bounds
        of the data (i.e. not padding). Note virtual dims are always in-bound.
    overlap_mask  : da.Array
        Dask Array object indicating which elements are part of an overlapped
        chunk (True), vs part of the original center chunk (False).
    dims          : Collection[str]
        Dimension names, ordered the same as the `data` axes.
    original_dims : Collection
        Contains original dimension ordering as well as the features coordinate
        vector. Todo: modify this to pass the information in more cleanly.
    resolution    : Collection[Number]
        Resolution for each dimension, ordered the same as the `data` axes.
    window_depth  : dict[str, np.ndarray[Int]]
        Format of {Dimension: (lower, upper)}, where (lower, upper) defines the
        number of elements to the left and to the right of the window center. 
        {'time': (1, 0)} would indicate a window which has two elements along
        the time dimension, where one element is the center and the other is 
        a single lookback step (to the left).
    valid_percent : dict[tuple[str], Number]
        Format of {(Dimension,): percent}, where the key is a tuple with at
        least one element; multiple dimensions in the key tuple indicate that
        the valid percent is applied across all of those dimensions combined.
        Percent is a value in the range [0, 1], indicating the percentage of 
        a window (for the given dimension(s)) which needs to be valid in order
        for the window itself to be valid. For example, 
        {('latitude', 'longitude'): 0.8, ('time',): 1} would indicate that for
        a given window, at least 80% of the elements must be valid across the 
        2d grid of latitude x longitude, and subsequently all elements across
        the time dimension must be valid. Note that the dictionary is ordered,
        and so dimensions are evaluated in the order given.
    invalid_value : Collection[Number]
        Collection of values which should be treated as NaN in the `data`.
    allow_repeats : bool
        If True, allow block overlap regions to be considered when generating 
        valid window centers. With this as True, it means that all possible 
        samples should be generated, but there is the potential for duplicate 
        samples to be generated across different blocks at the boundaries. If 
        instead this is False (the defalt), there will not be any duplicated
        samples generated, but instead there *may* be samples missing at the
        block boundaries. 

    """
    def __init__(self, 
        data          : da.Array,
        coords        : da.Array,
        inbound_mask  : da.Array,
        overlap_mask  : da.Array,
        resolution    : Union[Collection, da.Array],
        dims          : Collection[str],
        original_dims : Sequence,
        window_depth  : dict[str, np.ndarray]    = {},#dict[str, np.ndarray[Int]] = {},
        valid_percent : dict[tuple[str], Number] = {},
        invalid_value : Collection[object]       = [],
        allow_repeats : bool                     = False,
        block_count   : int                      = 1,
        block_index   : int                      = 0,
        label         : str                      = '',
    ):
        self._data    = data
        self._coords  = coords
        self._inbound = inbound_mask
        self._overlap = overlap_mask
        self.dims     = dims
        self.label    = label
        self.original_dims = original_dims
        self._resolution   = resolution
        self.window_depth  = window_depth
        self.valid_percent = valid_percent
        self.invalid_value = invalid_value
        self.allow_repeats = allow_repeats
        self.block_count   = block_count
        self.block_index   = block_index

        # Shape sanity checks
        arrays = [data, coords, inbound_mask, overlap_mask]
        shapes = [v.shape[:-1] for v in arrays]
        assert(len(set(shapes)) == 1), shapes
        assert(set(map(len, shapes)) == {len(dims)}), [shapes, dims]


    def __repr__(self) -> str:
        return f'Block[{self.label}]'


    @cached_property
    def benchmark(self):
        """ Return a Stopwatch function for benchmarking """
        debug = print if not hasattr(self, 'logger') else self.logger.debug
        return lambda label, logger=debug, **kwargs: Stopwatch(**({
            'message' : f'\t\t\t{self}.{label}',
            'logger'  : logger,
            'silent'  : not hasattr(self, 'logger'),
        } | kwargs))


    @cached_property
    def data(self) -> np.ndarray:
        """ Only compute dask data array upon first use """
        with self.benchmark(f'data {self._data.shape}'):
            return self._data.compute()


    @cached_property
    def coords(self) -> np.ndarray:
        """ Only compute dask coords array upon first use """
        with self.benchmark(f'coords {self._coords.shape}'):
            return self._coords.compute()


    @cached_property
    def inbound_mask(self) -> np.ndarray:
        """ Only compute dask mask array upon first use """
        return self._inbound.compute()


    @cached_property
    def overlap_mask(self) -> np.ndarray:
        """ Only compute dask mask array upon first use """
        return self._overlap.compute()


    @cached_property
    def resolution(self) -> list:
        """ Resolutions only need computed when non-uniform """
        return [getattr(r, 'compute', lambda: r)() for r in self._resolution]


    @cached_property
    def dtype(self) -> np.dtype:
        """ Create a composite datatype based on shapes of the data windows """
        sizes = {'features': self._data.shape[-1]} | self.window_total
        return np.dtype(
            [('values', self._data.dtype, tuple(sizes.values()))] + 
            [('coords', np.dtype([
                (dim, self._coords.dtype, (size,))
                for dim, size in sizes.items() ])
        )] )


    @property
    def shape(self) -> tuple[int]:
        """ Shape for all dimensions, excluding the feature dimension """
        return self._coords.shape[:-1]


    @property
    def ndim(self) -> int:
        """ Number of dimensions, excluding the feature dimension """
        return len(self.dims)


    @property
    def axes(self) -> dict[str, Int]:
        """ {Dimension key: axis number} """
        return {dim: i for i, dim in enumerate(self.dims)}
    

    @cached_property
    def window_total(self) -> dict[str, Int]:
        """ Window total size per dimension """
        return {dim: 1+d.sum() for dim, d in self.window_depth.items()}


    @cached_property
    def valid_windows_sparse(self):
        """ Determine valid sample window indices for sparse data """
        # Note that the sparse API isn't fully solidified - there are a few
        # different ways to handle extracting the valid windows (e.g. using
        # the sparse vectors explicitly; or using the COO interface into the
        # sparse array, similar to the dense calculation). Need to perform
        # some timing tests to see if using an API equivalent to dense (as 
        # is currently implemented) results in any slowdown / memory issues. 

        def window_count(valid: np.ndarray, keys_pct: tuple) -> np.ndarray: 
            """ Calculate the number of elements in the N-d rolling window """
            keys, percent = keys_pct
            
            # Get the required number of elements for a valid window
            n_total = np.prod([self.window_total[k] for k in keys])
            windows = [slice(-s, e+1) for e, s in self.window_depth.values()]

            # Expand the indices to be broadcastable with the N-d window
            idxs = self.data.coords[[self.axes[k] for k in keys]][:, valid]
            view = np.expand_dims(idxs, tuple(-(1+np.arange(len(windows)))))
            view = view + np.mgrid[tuple(windows)][:, None] # Broadcast window
            view = view.reshape(len(view), -1)              # (dims, samples)

            # Count the number of times an index appears and return valid ones
            view, counts = np.unique(view, axis=1, return_counts=True)
            valid_window = view[:, counts >= (percent * n_total)][:, None]
            valid[valid] = (idxs[..., None] == valid_window).all(0).any(-1)
            return valid 

        # return idxs[ counts >= (percent * n_total) ]
        # Create a mask indicating valid elements, and extract indices 
        data  = self.data.data 
        valid = ~self.invalid(data)
        valid = reduce(window_count, self.valid_percent.items(), valid)

        # Remove indices outside of coordinate bounds (except virtual)
        inbound = np.isfinite(self.coords)
        inbound|= np.isnan(self.coords).all(tuple(range(self.ndim)))
        return self.data.coords[:, valid & inbound.all(-1).flatten()][:-1].T
        # return valid & inbound.all(-1).flatten()


    @cached_property
    def valid_windows(self) -> np.ndarray:#[Int]:
        """ Determine indices for all valid sample windows in the block """
        if self.sparse: return self.valid_windows_sparse

        def moving_sum(invalid: np.ndarray, keys_pct: tuple) -> np.ndarray:
            """ Fast moving sum using bottleneck """
            if invalid.all(): return invalid

            keys, percent = keys_pct
            
            n_total = np.prod([self.window_total.get(k, 1) for k in keys])
            maximum = int((1-percent) * n_total)

            # Multiple axes can be used to define a combined valid percent 
            # e.g. {(ax1, ax2): 0.8} -> combination of the two must have 80% valid
            for exp, key in enumerate(keys):
                axis = self.axes[key]
                size = self.window_total.get(key, 1)

                if size > invalid.shape[axis]:
                    raise Exception(f'{self}: Block dimension "{key}" with ' +
                        f'size={invalid.shape[axis]} is too small for ' +
                        f'requested window size={size}')

                # Calculate window offets and padding
                offsets = (slice(None),) * axis + (slice(size - 1, None),)
                padding = [(0,0)] * invalid.ndim 
                padding[axis] = self.window_depth.get(key, (0,0))

                # Pad the invalid mask in order to center the window
                invalid = invalid.astype('int32') # Bottleneck is faster w/ int
                invalid = np.pad(invalid, padding, constant_values=size ** exp)

                # Offset the new invalid elements so that the window is centered
                invalid = bn.move.move_sum(invalid, size, axis=axis)[offsets]
            return invalid > maximum

        # Create mask indicating invalid elements, checking shortcuts first
        total_percent = np.round(np.prod(list(self.valid_percent.values())), 5)

        # If some elements need to be valid, we need to check for valid windows
        if total_percent > 0:
            # We need to mask features independently before collapsing the axis
            invalid = self.invalid()

            # Unless 100% need to be valid, in which case we can collapse now
            if total_percent == 1: invalid = invalid.any(-1, keepdims=True)

            # Calculate the running invalid mask over all dimensions
            invalid = reduce(moving_sum, self.valid_percent.items(), invalid)

        # Otherwise, checking is unnecessary since 0% are required to be valid
        else: invalid = np.zeros_like(self.inbound_mask)

        # Mask overlapped elements, as they cannot be window centers
        if not self.allow_repeats:
            if self.block_index == 0: invalid |= self.overlap_mask

        # Mask elements outside the bounds of the data (except virtual)
        invalid |= ~self.inbound_mask

        # Mask elements whose window extends outside of the data
        # Note that continuing if percent == 0 implies NaNs outside of the data
        #   are treated as always invalid, compared to NaNs in the data itself
        for keys, percent in self.valid_percent.items():
            if percent == 0: continue

            for dim in keys:
                offset = (slice(None),) * self.axes[dim]
                lo, hi = self.window_depth[dim]
                hi = invalid.shape[self.axes[dim]] - hi
                invalid[offset + (slice(None, lo),)] = True
                invalid[offset + (slice(hi, None),)] = True

        # Collapse feature dimension, since all features must be valid
        return np.array(np.where((~invalid).all(-1)), dtype='int32')


    @cached_property
    def valid_coords(self) -> np.ndarray:
        """ Coordinate values for the valid locations """
        # if self.sparse:
            # return self.coords.data.reshape(-1, self.coords.shape[-1])[self.valid_windows]
        return self.coords[tuple(self.valid_windows)]


    @cached_property
    def valid_data(self) -> np.ndarray:
        """ Center data values for the valid locations """
        # if self.sparse:
        #     return self.data.data.reshape(-1, self.data.shape[-1])[self.valid_windows]
        return self.data[tuple(self.valid_windows)]


    @cached_property
    def valid_resolution(self) -> np.ndarray:
        """ Resolution for valid locations, parsing left/right if necessary """
        if self.is_uniform: 
            return np.array(self.resolution)

        res = [r[v] for r,v in zip(self.resolution, self.valid_windows)]
        return np.stack(res, axis=1)


    @cached_property
    def fast_invalid_check(self) -> bool:
        """ Quick verification: returns True if no valid data exists """
        # If a dimension can have 0% exist and still be valid, don't check
        if min(self.valid_percent.values()) > 0:
            return self.invalid( self._data[..., 0].compute() ).all()
        return False


    @property
    def sparse(self) -> bool:
        """ Return True if data is a sparse object """
        return hasattr(type(self._data._meta), 'todense')


    @property
    def is_uniform(self) -> bool:
        """ Return True if the coordinate grid is uniform """
        return not isinstance(self._resolution[0], da.Array)


    def set_valid_percent(self, valid_percent: dict):
        """ Set a new valid_percent after formatting correctly """
        assert(getattr(self, '_original_valid_percent', None) is None)
        self._original_valid_percent = self.valid_percent
        formatted = {}
        keys = tuple()
        for k, v in valid_percent.items():
            if not isinstance(k, tuple):
                k = (k,)
            keys += k
            formatted[k] = v
        for d in self.dims:
            if d not in keys:
                formatted[(d,)] = 1
        self.valid_percent = formatted


    def reset_valid_percent(self):
        if getattr(self, '_original_valid_percent', None) is not None:
            self.valid_percent = self._original_valid_percent
            self._original_valid_percent = None


    def invalid(self, data: Union[np.ndarray, None] = None):
        """Element-wise mask of invalid values in the given array.

        Parameters
        ----------
        data : np.ndarray | None
            The array to create a mask for. If no parameter is given, 
            self.data is used (the full block data).

        Returns
        -------
        np.ndarray
            Boolean mask which indicates where there are invalid values.

        """
        data = (data if data is not None else self.data)

        # Need to cast to object if either array is not numeric
        if not np.issubdtype(data.dtype, np.number):
            invalid = np.array(self.invalid_value, dtype=object)
        else: invalid = np.array(self.invalid_value)

        # Numpy also converts something like [1,'a'] to ['1','a']
        # so we need to recast invalid as an object dtype in case
        if not np.issubdtype(invalid.dtype, np.number):
            invalid = np.array(self.invalid_value, dtype=object)
            data = data.astype(object)

        # np.isnan does not work on object dtype
        # with pd.option_context('use_inf_as_na', True):
        return pd.isna(data) | np.isin(data, invalid)


    def cleanup(self):
        """ Delete cached objects to free memory; uncertain whether
            this is actually necessary, and may cause minor slowdowns """
        for key in ['valid_coords', 'valid_windows', 'data', 'coords']:
            if key in self.__dict__:
                del self.__dict__[key]

    
    def extract(self, matches: np.ndarray, return_features: list[str] | None = None, empty=False):# -> dict[Int, xr.Dataset]:
        """Extract a list of windows from data, wrapping each with xarray.

        Parameters
        ----------
        matches : np.ndarray
            Indices for valid_windows, which indicates which window locations
            are being used for this (eventually created) SampleSet. 

        Returns
        -------
        list[xr.Dataset]
            Returns a list of the extracted xr.Dataset windows, where each
            one corresponds to a location taken from data/coords, and the 
            size is defined by the original window_depth. Length of the 
            returned list equals the length of the input `indices`.

        """
        # If we already know the set of features needed, just return that array
        if return_features is not None:
            orig_dims, features, dtypes, req_features = self.original_dims
            orig_dims = ['features'] + [d for d in orig_dims if d != 'features']
            keep_dims = [f for f in self.dims if f in orig_dims] + ['features']

            # Extract only the requested features to be returned
            available = np.array(list(features) + list(orig_dims)[1:])
            requested = [f.split('@')[0] for f in return_features]
            
            # Cast dtypes for each feature later (to avoid mixed type array)
            mask = np.isin(available, requested)
            keys = dict(zip(requested, return_features))
            dtypes = {keys[k]: dtypes[k] for k in available[mask]}

            # Create arrays for the left and total window depths
            lower = np.array([self.window_depth[k][0]    for k in self.dims])
            total = np.array([self.window_total.get(k,1) for k in self.dims])
            shape = tuple(s for s,d in zip(total, self.dims) if d in orig_dims)
            ndims = len(self.dims)

            # If this was a dropped Block, return just a NaN array placeholder
            if empty: return np.nan*np.zeros((len(requested),)+shape, 'float32'), dtypes
            
            # Otherwise, get all unique match indices
            unique, indices = np.unique(matches, return_inverse=True)

            # Calculate the lower and upper bounds for each window dimension
            center = np.array(self.valid_windows)[:, unique, None]
            center-= lower[:, None, None]
            bounds = [left + np.arange(size) for left, size in zip(center, total)]

            # Remove virtual dimension from coordinate features
            real_dims = sorted(map(self.dims.index, orig_dims[1:]))
            n_samples = len(unique)

            def expand(axis: int, bound: np.ndarray) -> np.ndarray:
                """ Add dimensions to each bound based on the dim it applies to """
                return np.expand_dims(bound, list(set(range(1, ndims+1)) - {axis}))

            # Expand the bounds so they can be broadcast over the full data/coords
            windows = tuple(starmap(expand, enumerate(bounds, 1)))
            assert(len(windows[0]) == n_samples), [len(windows), n_samples]

            # Extract windows and drop virtual dimensions 
            # Note: SegFault/Access violation here is likely an issue with data
            #       that is cached on disk; try removing the cache to resolve.
            data = np.append(self.data, self.coords[..., real_dims], axis=-1)
            data = data[..., mask][windows].reshape((n_samples,) + shape + (len(requested),))

            # Transpose data to the correct order: [samples, features, ...]
            data = data.transpose(tuple(map((['']+keep_dims).index, ['']+orig_dims)))
            return data[indices], dtypes

        orig_dims, features, dtypes, req_features = self.original_dims
        orig_dims = ['features'] + [d for d in orig_dims if d != 'features']

        # Create arrays for the left and total window depths
        lower = np.array([self.window_depth[k][0]    for k in self.dims])
        total = np.array([self.window_total.get(k,1) for k in self.dims])
        ndims = len(self.dims)
        shape = tuple(s for s,d in zip(total, self.dims) if d in orig_dims)

        # Initialize the final result dict with all globally applicable values
        keep_dims = [f for f in self.dims if f in orig_dims] + ['features']
        xr_kwargs = {'dims': orig_dims, 'requested_features': req_features} | (
            {'attrs' : {'resolution': dict(zip(self.dims, self.resolution))}}
            if self.is_uniform else {})
        
        def cast_dtype(key: str, value: np.ndarray) -> np.ndarray:
            """ Cast the given value array back to its original dtype """
            return value.astype(dtypes[key])

        def expand(axis: int, bound: np.ndarray) -> np.ndarray:
            """ Add dimensions to each bound based on the dim it applies to """
            return np.expand_dims(bound, list(set(range(1, ndims+1)) - {axis}))

        def collapse(axis: int, coord: np.ndarray) -> np.ndarray:
            """ Collapse the coord grid into its respective coord vector """
            return coord[(0,)*(coord.ndim-axis-2)+(slice(None),)+(0,)*axis].T

        def gen_xr_dict(data: np.ndarray, *coords: np.ndarray) -> dict:
            """ Generate the xr.Dataset dict for given data/coord windows """            
            return { 'data'   : list(map(cast_dtype, features, data)), 
                     'coords' : dict(zip(keep_dims, coords)) | {
                        'features': features} } | xr_kwargs

        # If this was a dropped Block, use just NaN array placeholders
        if empty:
            data = np.nan * np.zeros((len(features),)+shape, 'float32')
            coords = [np.nan * np.zeros(s, 'float32') for s in shape]
            return [gen_xr_dict(data, *coords)]

        ravel   = lambda v: getattr(v, 'ravel', lambda: v)()
        indices = tuple(np.unique([j for i in matches for j in ravel(i)]))
        samples = len(indices)

        # Calculate the lower and upper bounds for each window dimension
        center = np.array(self.valid_windows)[:, indices, None]
        center-= lower[:, None, None]
        bounds = [left + np.arange(size) for left, size in zip(center, total)]

        # Expand the bounds so they can be broadcast over the full data/coords
        windows = tuple(starmap(expand, enumerate(bounds, 1)))
        assert(len(windows[0]) == samples), [len(windows), samples]

        # Extract windows and drop virtual dimensions 
        # Note: SegFault/Access violation here is likely an issue with data
        #       that is cached on disk; try removing the cache to resolve.
        data   =   self.data[windows].reshape((samples,) + shape + (len(features),))
        coords = self.coords[windows].reshape((samples,) + shape + (len(self.dims),))
        
        # Remove virtual dimension from coordinate features
        coords = coords[..., sorted(map(self.dims.index, orig_dims[1:]))]

        # Transpose data to the correct order: [samples, features, ...]
        data = data.transpose(tuple(map((['']+keep_dims).index, ['']+orig_dims)))

        # Extract coordinate vectors
        coord_vectors = starmap(collapse, enumerate(coords.T))
        coord_vectors = list(map(cast_dtype, keep_dims, coord_vectors))

        # Extract final window dictionaries to use for Sample initialization 
        windows = dict(zip(indices, map(gen_xr_dict, data, *coord_vectors)))
        return [[windows[i] for i in np.atleast_1d(idx)] for idx in matches]


    def feature_subset(self, features: Sequence | None) -> list[str] | None:
        """ Return a list of the features that can be provided by this Block.
        
        Parameters
        ----------
        features : Sequence | None
            A (possibly nested) collection of string features that compose
            the complete set of features being requested for loading. The
            nesting structure defines how features should be organized once
            returned by the pipeline, but can be ignored by this function
            aside from the un-nesting process. See the `features` keyword
            docstring in `Batcher` for additional information on nesting.
        
        Returns
        -------
        list[str]
            The subset of features that this Block can provide, returned as
            a (flat) list of strings.
        
        """

        # Features being None means Sample objects will be created, so ignore
        if features is not None:
            available_features = sum(map(list, self.original_dims[:2]), [])
            available_features = list(set(available_features)-{'features'})

            def recurse(feature: Sequence | str) -> list:
                """ Recurse over the nested structure to find availablity """

                # Feature is still nested if it's not a string
                if not isinstance(feature, str):
                    return sum(map(recurse, feature or available_features), [])

                # Also need to handle @{index} syntax, to allow feature
                # selection by Datafile when there are duplicated features
                name, i, *extra = (feature+f'@{self.block_index}').split('@')

                if int(i) == self.block_index:
                    if name in available_features: 
                        return [feature]

                    # If any indices remain, it means this block was explicitly
                    # requested to provide this feature - which it doesn't have
                    assert(len(extra)==0), f'{feature} unavailable from {self}'
                return []
            return list(set( recurse(features or available_features) ))