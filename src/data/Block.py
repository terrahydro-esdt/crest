from collections.abc import Collection
from functools import cached_property, reduce
from itertools import starmap
from numbers import Number, Integral as Int

import dask.array as da 
import bottleneck as bn
import xarray as xr
import numpy as np 

from crest.src.base import BaseAbstract 



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
        invalid_value : Collection[Number]         = [],
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
        sizes = {'features': self.data.shape[-1]} | self.window_total
        return np.dtype(
            [('values', np.float32, tuple(sizes.values()))] + 
            [('coords', np.dtype(
                [(dim, np.float32, (size,)) for dim, size in sizes.items()])
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
    def valid_windows(self) -> np.ndarray[Int]:
        """ Determine indices for all valid sample windows in the block """
        
        def moving_sum(invalid: np.ndarray, key: str) -> np.ndarray:
            """ Fast moving sum using bottleneck """
            axis = self.axes[key]
            size = self.window_total[key]
            offsets = [slice(None)] * axis + [slice(size - 1, None)]
            invalid = bn.move.move_sum(invalid, size, axis=axis)
            return invalid[tuple(offsets)]

        # Create a boolean mask indicating invalid elements
        invalid = np.isnan(self.data) | np.isin(self.data, self.invalid_value)

        # Calculate the running invalid mask by iterating over valid percents
        for keys, percent in self.valid_percent.items():            
            n_total = np.prod([self.window_total[k] for k in keys])
            maximum = int((1-self.valid_percent[keys]) * n_total)
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
        return self.coords[tuple(self.valid_windows)]


    @cached_property
    def valid_data(self) -> np.ndarray:
        """ Retrieve the center data values for the valid locations """
        return self.data[tuple(self.valid_windows)]


    def extract(self, indices: np.ndarray) -> list[xr.Dataset]:
        """Extract a list of windows from data, wrapping each with xarray.

        Parameters
        ----------
        indices : np.ndarray
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
        ndims = len(self.dims)
        lower = np.array([self.window_depth[k][0] for k in self.dims])
        total = np.array([self.window_total[k]    for k in self.dims])

        # Calculate the lower and upper bounds for each window dimension
        center = np.array(self.valid_windows)[:, tuple(indices), None]
        center-= lower[:, None, None]
        bounds = [left + np.arange(size) for left, size in zip(center, total)]

        original_dims, features = self.original_dims

        def expand(axis: int, bound: np.ndarray) -> np.ndarray:
            """ Add dimensions to each bound based on the dim it applies to """
            return np.expand_dims(bound, list(set(range(1, ndims+1)) - {axis}))
        
        def collapse(axis: int, coord: np.ndarray) -> np.ndarray:
            """ Collapse the coord grid into its respective coord vector """
            return coord[(0,)*(coord.ndim-axis-1) + (slice(None),) + (0,)*axis]

        def gen_coords(coords: np.ndarray) -> dict[str, np.ndarray]:
            """ Generate a coordinates vector dictionary for xarray """
            return dict(zip(self.dims, starmap(collapse, enumerate(coords.T))))

        def gen_dataset(data: np.ndarray, coords: np.ndarray) -> xr.Dataset:
            """ Generate the xr.Dataset for the given data/coord windows """
            attributes = {'resolution': dict(zip(self.dims, self.resolution))}
            return xr.DataArray(data, **{
                'coords' : gen_coords(coords) | {'features': features}, 
                'dims'   : self.dims + ['features'],
                'attrs'  : attributes,
            }).transpose(*original_dims).to_dataset('features')

        # Expand the bounds so they can be broadcast over the full data/coords
        windows = tuple(starmap(expand, enumerate(bounds, 1)))
        return list(map(gen_dataset, self.data[windows], self.coords[windows]))



