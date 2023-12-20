from collections.abc import Collection, Callable
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

    """
    def __init__(self, 
        data          : da.Array,
        coords        : da.Array,
        inbound_mask  : da.Array,
        overlap_mask  : da.Array,
        resolution    : Union[Collection, da.Array],
        dims          : Collection[str],
        original_dims : Collection,
        window_depth  : dict[str, np.ndarray] = {},#dict[str, np.ndarray[Int]] = {},
        valid_percent : dict[tuple[str], Number]   = {},
        invalid_value : Collection[object]         = [],
        block_count   : int                        = 1,
        block_index   : int                        = 0,
        label         : str                        = '',
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
        return lambda label, log=debug: Stopwatch(
            prefix=f'\t\t\t{self}.{label}',
            logger=log,
            silent=not hasattr(self, 'logger'),
        )


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
        total_percent = round(sum(self.valid_percent.values()), 5)

        # If 0% of elements need to be valid, we can just return the indices
        if total_percent == 0:
            return np.indices(self.shape).reshape((self.ndim, -1))

        # If all elements in window must be valid, we can collapse the features
        elif total_percent >= len(self.valid_percent):
            invalid = self.invalid().any(-1, keepdims=True)

        # Otherwise, we need to mask independently before collapsing features
        else: invalid = self.invalid()

        # Calculate the running invalid mask over all dimensions
        invalid = reduce(moving_sum, self.valid_percent.items(), invalid)

        # Mask overlapped elements, as they cannot be window centers
        if self.block_index == 0: 
            invalid |= self.overlap_mask

        # Mask elements outside the bounds of the data (except virtual)
        invalid |= ~self.inbound_mask

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
    def fast_invalid_check(self):
        """ Quick check to verify there exists any valid data """
        return self.invalid( self._data[..., 0].compute() ).all()


    @property
    def sparse(self) -> bool:
        """ Return True if data is a sparse object """
        return hasattr(type(self._data._meta), 'todense')


    @property
    def is_uniform(self) -> bool:
        """ Return True if the coordinate grid is uniform """
        return not isinstance(self._resolution[0], da.Array)


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
        with pd.option_context('use_inf_as_na', True):
            return pd.isna(data) | np.isin(data, invalid)


    def cleanup(self):
        """ Delete cached objects to free memory; uncertain whether
            this is actually necessary, and may cause minor slowdowns """
        for key in ['valid_coords', 'valid_windows', 'data', 'coords']:
            if key in self.__dict__:
                del self.__dict__[key]

    
    def extract(self, matches: np.ndarray):# -> dict[Int, xr.Dataset]:
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
        ravel   = lambda v: getattr(v, 'ravel', lambda: v)()
        indices = tuple(np.unique([j for i in matches for j in ravel(i)]))
        samples = len(indices)

        # Create arrays for the left and total window depths
        lower = np.array([self.window_depth[k][0]    for k in self.dims])
        total = np.array([self.window_total.get(k,1) for k in self.dims])
        ndims = len(self.dims)

        # Calculate the lower and upper bounds for each window dimension
        center = np.array(self.valid_windows)[:, indices, None]
        center-= lower[:, None, None]
        bounds = [left + np.arange(size) for left, size in zip(center, total)]

        orig_dims, features = self.original_dims
        orig_dims = ['features'] + [d for d in orig_dims if d != 'features']

        # Initialize the final result dict with all globally applicable values
        keep_dims = [f for f in self.dims if f in orig_dims] + ['features']
        xr_kwargs = {'dims': orig_dims} | ({
            'attrs' : {'resolution': dict(zip(self.dims, self.resolution))}
        } if self.is_uniform else {})

        def expand(axis: int, bound: np.ndarray) -> np.ndarray:
            """ Add dimensions to each bound based on the dim it applies to """
            return np.expand_dims(bound, list(set(range(1, ndims+1)) - {axis}))
        
        def collapse(axis: int, coord: np.ndarray) -> np.ndarray:
            """ Collapse the coord grid into its respective coord vector """
            return coord[(0,)*(coord.ndim-axis-1) + (slice(None),) + (0,)*axis]

        def gen_coords(coords: np.ndarray) -> dict[str, np.ndarray]:
            """ Generate a coordinates vector dictionary for xarray """
            coord_vectors = starmap(collapse, enumerate(coords.T))
            return dict(zip(keep_dims, coord_vectors)) | {'features': features}

        def gen_xr_dict(data: np.ndarray, coords: np.ndarray) -> dict:
            """ Generate the xr.Dataset dict for given data/coord windows """
            return xr_kwargs | {'data': data, 'coords': coords}

        # Expand the bounds so they can be broadcast over the full data/coords
        windows = tuple(starmap(expand, enumerate(bounds, 1)))
        assert(len(windows[0]) == samples), [len(windows), samples]

        # Extract windows and drop virtual dimensions 
        # Note: SegFault/Access violation here is likely an issue with data
        #       that is cached on disk; try removing the cache to resolve.
        shapes = tuple(s for s,d in zip(total, self.dims) if d in orig_dims)
        data   =   self.data[windows].reshape((samples,) + shapes + (len(features),))
        coords = self.coords[windows].reshape((samples,) + shapes + (len(self.dims),))
        
        # Remove virtual dimension from coordinate features
        coords = coords[..., sorted(map(self.dims.index, orig_dims[1:]))]

        # Transpose data to the correct order (offset by 1)
        data = data.transpose(tuple(map((['']+keep_dims).index, ['']+orig_dims)))

        # Extract final window dictionaries to use for Sample initialization 
        coords  = map(gen_coords, coords)
        windows = dict(zip(indices, map(gen_xr_dict, data, coords)))
        return [[windows[i] for i in np.atleast_1d(idx)] for idx in matches]
