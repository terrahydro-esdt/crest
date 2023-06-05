from collections.abc import Collection
from functools import cached_property, reduce
from itertools import starmap
from numbers import Number, Integral as Int

import dask.array as da 
import bottleneck as bn
import xarray as xr
import pandas as pd
import numpy as np 
import sparse 

from crest.src.base import BaseAbstract 
from crest.src.utils import Stopwatch



class Block(BaseAbstract):
    """Class which wraps a dask block. 

    Parameters
    ----------
    data          : da.Array
        Dask Array object containing the data within the block.
    coords        : da.Array
        Dask Array object containing the coordinates which define the block.
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
        dims          : Collection[str],
        original_dims : Collection,
        resolution    : Collection[Number],
        window_depth  : dict[str, np.ndarray[Int]] = {},
        valid_percent : dict[tuple[str], Number]   = {},
        invalid_value : Collection[object]         = [],
    ):
        self._data   = data
        self._coords = coords
        self.dims    = dims
        self.original_dims = original_dims
        self.resolution    = resolution
        self.window_depth  = window_depth
        self.valid_percent = valid_percent
        self.invalid_value = invalid_value


    @cached_property
    def data(self) -> np.ndarray:
        """ Only compute dask data array upon first use """
        return self._data.compute()


    @cached_property
    def coords(self) -> np.ndarray:
        """ Only compute dask coords array upon first use """
        return self._coords.compute()


    @cached_property
    def dtype(self) -> np.dtype:
        """ Create a composite datatype based on shapes of the data windows """
        sizes = {'features': self._data.shape[-1]} | self.window_total
        return np.dtype(
            [('values', self._data.dtype, tuple(sizes.values()))] + 
            [('coords', np.dtype(
                [(dim, self._coords.dtype, (size,)) for dim, size in sizes.items()])
        )] )


    @cached_property
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

        def window_count(valid: np.ndarray, keys_pct: tuple): 
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
        inbound|= np.isnan(self.coords).all(tuple(range(self.coords.ndim - 1)))
        return self.data.coords[:, valid & inbound.all(-1).flatten()][:-1].T
        # return valid & inbound.all(-1).flatten()


    @cached_property
    def valid_windows(self) -> np.ndarray[Int]:
        """ Determine indices for all valid sample windows in the block """
        if self.sparse: return self.valid_windows_sparse

        def moving_sum(invalid: np.ndarray, key: str) -> np.ndarray:
            """ Fast moving sum using bottleneck """
            axis = self.axes[key]
            size = self.window_total[key]
            offsets = [slice(None)] * axis + [slice(size - 1, None)]
            invalid = invalid.astype('int32') # Bottleneck has fast impl w/ int
            invalid = bn.move.move_sum(invalid, size, axis=axis)
            return invalid[tuple(offsets)]

        # Create a boolean mask indicating invalid elements
        invalid = self.invalid()

        # Calculate the running invalid mask by iterating over valid percents
        for keys, percent in self.valid_percent.items():   
            if (~invalid).any():         
                n_total = np.prod([self.window_total[k] for k in keys])
                maximum = int((1-percent) * n_total)
                invalid = reduce(moving_sum, keys, invalid) > maximum

        # Offset the final mask indices in order to center the window
        offset  = np.array([[self.window_depth[dim][0]] for dim in self.dims])
        indices = np.array(np.where((~invalid).all(-1))) + offset

        # Remove indices outside of coordinate bounds (except virtual)
        inbound = np.isfinite( self.coords[tuple(indices)] )
        inbound|= np.isnan(self.coords).all(tuple(range(self.coords.ndim - 1)))
        return indices[:, inbound.all(1)]


    @cached_property
    def valid_coords(self) -> np.ndarray:
        """ Retrieve the coordinate values for the valid locations """
        # if self.sparse:
            # return self.coords.data.reshape(-1, self.coords.shape[-1])[self.valid_windows]
        return self.coords[tuple(self.valid_windows)]


    @cached_property
    def valid_data(self) -> np.ndarray:
        """ Retrieve the center data values for the valid locations """
        # if self.sparse:
        #     return self.data.data.reshape(-1, self.data.shape[-1])[self.valid_windows]
        return self.data[tuple(self.valid_windows)]


    @cached_property
    def fast_invalid_check(self):
        """ Quick check to verify there exists any valid data """
        return self.invalid( self._data[..., 0].compute() ).all()


    @property
    def sparse(self):
        """ Return True if data is a sparse object """
        return hasattr(type(self._data._meta), 'todense')


    def invalid(self, data: np.ndarray | None = None):
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
        # for key in ['valid_coords', 'valid_windows', 'data', 'coords']:
        #     if key in self.__dict__:
        #         del self.__dict__[key]

    
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
        indices = tuple(np.unique([j for i in matches for j in i.ravel()]))

        ndims = len(self.dims)
        lower = np.array([self.window_depth[k][0] for k in self.dims])
        total = np.array([self.window_total[k]    for k in self.dims])

        # Calculate the lower and upper bounds for each window dimension
        center = np.array(self.valid_windows)[:, indices, None]
        center-= lower[:, None, None]
        bounds = [left + np.arange(size) for left, size in zip(center, total)]

        original_dims, features = self.original_dims
        original_dims = ['features'] + [d for d in original_dims if d != 'features']

        full_dims = self.dims + ['features']
        dim_order = np.array(list(map(full_dims.index, original_dims))) + 1
        xr_kwargs = {
            'dims'  : [d for d in original_dims if self.window_total.get(d, 2) > 1],
            'attrs' : {'resolution': dict(zip(self.dims, self.resolution))},
            'order' : original_dims,
        }

        def expand(axis: int, bound: np.ndarray) -> np.ndarray:
            """ Add dimensions to each bound based on the dim it applies to """
            return np.expand_dims(bound, list(set(range(1, ndims+1)) - {axis}))
        
        def collapse(axis: int, coord: np.ndarray) -> np.ndarray:
            """ Collapse the coord grid into its respective coord vector """
            return coord[(0,)*(coord.ndim-axis-1) + (slice(None),) + (0,)*axis]

        def gen_coords(coords: np.ndarray) -> dict[str, np.ndarray]:
            """ Generate a coordinates vector dictionary for xarray """
            return dict(zip(self.dims, starmap(collapse, enumerate(coords.T)))) | {'features': features}

        def gen_dataset(data: np.ndarray, coords: np.ndarray):# -> xr.Dataset:
            """ Generate the xr.Dataset for the given data/coord windows """
            return xr_kwargs | {'data': data, 'coords': coords}
            # return xr.DataArray(data, coords, **xr_kwargs).to_dataset('features')

        # Expand the bounds so they can be broadcast over the full data/coords
        windows = tuple(starmap(expand, enumerate(bounds, 1)))
        assert(len(windows[0]) == len(indices)), [len(windows), len(indices)]

        # Transpose data to be in the correct order, and extract final windows
        data    = self.data[windows].transpose((0,)+tuple(dim_order))
        data    = data.reshape(data.shape[:2] + tuple(s for s in data.shape[2:] if s > 1))
        coords  = map(gen_coords, self.coords[windows])
        windows = dict(zip(indices, map(gen_dataset, data, coords)))
        return [[windows[i] for i in np.atleast_1d(match)] for match in matches]
