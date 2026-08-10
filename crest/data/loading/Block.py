from __future__ import annotations
from collections.abc import Collection, Sequence
from collections import defaultdict as dd
from functools import cached_property, reduce
from itertools import starmap, chain
from numbers import Number, Integral as Int
from typing import Union

import dask.dataframe as df
import dask.array as da
import bottleneck as bn
import xarray as xr
import pandas as pd
import numpy as np
import warnings
import dask

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
        data          : Union[da.Array, Collection],
        coords        : da.Array,
        inbound_mask  : da.Array,
        overlap_mask  : da.Array,
        valid_mask    : Union[None, da.Array]       = None,
        resolution    : Union[Collection, da.Array] = [],
        dims          : Collection[str]             = [],
        original_dims : Sequence                    = [],
        window_depth  : dict[str, np.ndarray]       = {},#dict[str, np.ndarray[Int]] = {},
        valid_percent : dict[tuple[str], Number]    = {},
        invalid_value : Collection[object]          = [],
        allow_repeats : bool  = False,
        block_count   : int   = 1,
        block_index   : int   = 0,
        label         : str   = '',
        sparsity      : float = 0.,
        coord_vecs    : Collection = [],
        key_label     : str | None = None
    ):
        self._data    = data
        self._coords  = coords
        self._inbound = inbound_mask
        self._overlap = overlap_mask
        self.dims     = dims
        self.label    = label
        self.sparsity = sparsity
        self.original_dims = original_dims
        self._resolution   = resolution
        self._valid_mask   = valid_mask
        self._window_depth = window_depth
        self.valid_percent = valid_percent
        self.invalid_value = invalid_value
        self.allow_repeats = allow_repeats
        self.block_count   = block_count
        self.block_index   = block_index
        self.coord_vecs = coord_vecs
        self.key_label = key_label

        # Shape sanity checks
        arrays = [coords, inbound_mask, overlap_mask]
        shapes = [v.shape[:-1] for v in arrays]
        assert(len(set(shapes)) == 1), shapes
        assert(set(map(len, shapes)) == {len(dims)}), [shapes, dims]

        # Sparse data should already be filtered so it contains only valid data
        # if self.is_sparse and len(self.invalid_value):
        #     raise Exception('Sparse data should already be filtered')


    def __repr__(self) -> str:
        return f'Block[{self.label}]'


    def copy(self):
        blk = Block(
            data = self._data,
            coords = self._coords,
            inbound_mask = self._inbound,
            overlap_mask = self._overlap,
            dims = self.dims,
            label = self.label,
            sparsity = self.sparsity,
            original_dims = self.original_dims,
            resolution = self._resolution,
            valid_mask = self._valid_mask,
            window_depth = self._window_depth,
            valid_percent = self.valid_percent,
            invalid_value = self.invalid_value,
            allow_repeats = self.allow_repeats,
            block_count = self.block_count,
            block_index = self.block_index,
            coord_vecs = self.coord_vecs,
            key_label = self.key_label,
        )
        blk.__dict__.update(self.__dict__)
        return blk


    @cached_property
    def benchmark(self):
        """ Return a Stopwatch function for benchmarking """
        debug = print if not hasattr(self, 'logger') else self.logger.debug
        return lambda label, logger=debug, **kwargs: Stopwatch(**({
            'message' : f'\t\t\t{self}.{label}',
            'logger'  : logger,
            'silent'  : not hasattr(self, 'logger'),
        } | kwargs))


    @property
    def is_required(self) -> bool:
        """ Whether this block is required to form a valid sample """
        return np.prod(list(self.valid_percent.values())) > 0


    @property
    def is_sparse(self) -> bool:
        """ Return True if data is a sparse object """
        data = self._data if isinstance(self._data,da.Array) else self._data[0]
        return hasattr(type(data._meta), 'todense')


    @property
    def is_uniform(self) -> bool:
        """ Return True if the coordinate grid is uniform """
        return not isinstance(self._resolution[0], da.Array)


    @property
    def features(self) -> list[str]:
        """ Return the list of available features """
        return self.original_dims[1]


    @property
    def data(self) -> np.ndarray:
        """ Only compute dask data array upon first use """
        with self.benchmark(f'data'):
            if isinstance(self._data, da.Array):
                return self._data.compute()
            return np.concatenate(da.compute(*[d for d in self._data]), axis=-1)


    @cached_property
    def sparse_data(self):
        return self.data


    @property
    def dataset(self) -> xr.Dataset:
        """ Create an xarray dataset from the data and coordinates """
        flat = [(self.dims, v[..., 0]) for v in self._data]
        data = dict(zip(self.features, flat))
        return xr.Dataset(data, coords=dict(zip(self.dims, self.coord_vecs)))


    @cached_property
    def coords(self) -> np.ndarray:
        """ Only compute dask coords array upon first use """
        if self.is_sparse and len(self.coord_vecs):
            index = np.isfinite(self.sparse_data).all(-1).coords
            # print(f'\t{self.data.shape=} {self.sparse_data.shape=} {self.sparse_data.coords.shape=} {[c.shape for c in self.coord_vecs]=} {index.shape=}')
            if not index.size: raise Exception(f'No index found: {self=}')#self.interactive()
            coord = [c[i].compute() for c, i in zip(self.coord_vecs, index)]
            return np.stack(coord, axis=-1)
        with self.benchmark(f'coords {self._coords.shape}'):
            return self._coords.compute()


    @property
    def inbound_mask(self) -> np.ndarray:
        """ Only compute dask mask array upon first use """
        return self._inbound.compute()


    @property
    def overlap_mask(self) -> np.ndarray:
        """ Only compute dask mask array upon first use """
        return self._overlap.compute()


    @property
    def resolution(self) -> list:
        """ Resolutions only need computed when non-uniform """
        return [getattr(r, 'compute', lambda: r)() for r in self._resolution]


    @cached_property
    def dtype(self) -> np.dtype:
        """ Create a composite datatype based on shapes of the data windows """
        if isinstance(self._data, da.Array):
            n_features = self._data.shape[-1]
            data_dtype = self._data.dtype
        else:
            n_features = len(self._data)
            data_dtype = self._data[0].dtype

        sizes = {'features': n_features} | self.window_total
        return np.dtype(
            [('values', data_dtype, tuple(sizes.values()))] +
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


    @property
    def window_depth(self) -> dict[str, (Int, Int)]:
        """ Window sizes for each dimension: {dim: (n_left, n_right)} """
        return {dim: self._window_depth.get(dim, (0,0)) for dim in self.dims}


    @property
    def window_total(self) -> dict[str, Int]:
        """ Total size for each dimension: {dim: 1 + n_left + n_right} """
        return {dim: 1+sum(d) for dim, d in self.window_depth.items()}


    @property
    def uses_window(self) -> bool:
        """ False if the requested window contains only a single element """
        return any(size > 1 for size in self.window_total.values())


    @property
    def valid_windows_sparse(self):
        """ Determine valid sample window indices for sparse data """
        # If no window is requested, we can just use the sparse coords directly
        if not self.uses_window:
            total_percent = np.round(np.prod(list(self.valid_percent.values())), 5)
            if total_percent >= (1-1e-4):
                return np.isfinite(self.sparse_data).all(-1).coords
            return (np.isfinite(self.sparse_data).mean(-1) >= total_percent).coords
            # return self.sparse_data.coords[:-1]

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
            idxs = self.sparse_data[...,0].coords[[self.axes[k] for k in keys]][:, valid]
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
        data  = self.sparse_data[..., 0]
        valid = ~self.invalid(data.data)
        valid = reduce(window_count, self.valid_percent.items(), valid)

        # Remove indices outside of coordinate bounds (except virtual)
        # inbound = np.isfinite(self.coords)
        # inbound|= np.isnan(self.coords).all(tuple(range(self.ndim)))
        depth = np.array([self.window_depth[k] for k in self.dims])
        lower = (data.coords.T - depth[:, 0]) < 0
        upper = (data.coords.T + depth[:, 1]) >= data.shape
        outer = (lower | upper).any(1)
        return valid & ~outer
        # return data.coords[:, valid][:-1]# & inbound.all(-1).flatten()][:-1].T
        # return valid & inbound.all(-1).flatten()


    @cached_property
    def valid_windows(self) -> np.ndarray:#[Int]:
        """ Determine indices for all valid sample windows in the block """
        if self.is_sparse: return self.valid_windows_sparse

        def moving_sum(invalid: np.ndarray, keys_pct: tuple) -> np.ndarray:
            """ Fast moving sum using bottleneck """
            if invalid.all(): return invalid

            keys, percent = keys_pct

            # Valid percent is handled in a way that allows a small percent
            # to be used in indicating 'at least one element'. For example,
            # in a 3x3 spatial window a valid percent of 0.01 could be used
            # to ensure at least one element in windows are valid. In other
            # words, percentages are interpreted as ceil(n_elements * pct).
            n_total = np.prod([self.window_total[k] for k in keys])
            maximum = int((1-percent) * n_total)

            # Multiple axes can be used to define a combined valid percent
            # e.g. {(ax1, ax2): 0.8} -> combination of the two must have 80% valid
            for exp, key in enumerate(keys):
                axis = self.axes[key]
                size = self.window_total[key]

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
            invalid = ~self.valid_mask#self.invalid()

            # Unless 100% need to be valid, in which case we can collapse now
            if total_percent == 1: invalid = invalid.any(-1, keepdims=True)

            # Calculate the running invalid mask over all dimensions
            invalid = reduce(moving_sum, self.valid_percent.items(), invalid)

        # Otherwise, checking is unnecessary since 0% are required to be valid
        else: invalid = np.zeros(self._inbound.shape, dtype='bool')

        # Mask overlapped elements, as they cannot be window centers
        if not self.allow_repeats:
            if self.block_index == 0: invalid |= self.overlap_mask

        # Mask elements outside the bounds of the data (except virtual)
        invalid |= ~self.inbound_mask

        if hasattr(self, '_feature_mask'):
            f_mask = self._feature_mask
            invalid |= ~getattr(f_mask, 'compute', lambda: f_mask)()

        # Mask elements whose window extends outside of the data
        # Note that continuing if percent == 0 implies NaNs outside of the data
        #   are treated as always invalid, compared to NaNs in the data itself
        for keys, percent in self.valid_percent.items():
            # if percent == 0: continue

            for dim in keys:
                offset = (slice(None),) * self.axes[dim]
                lo, hi = self.window_depth[dim]
                hi = invalid.shape[self.axes[dim]] - hi
                invalid[offset + (slice(None, lo),)] = True
                invalid[offset + (slice(hi, None),)] = True

        # Collapse feature dimension, since all features must be valid
        dtype = 'uint16' if max(invalid.shape) < 65634 else 'int32'
        return np.array(np.where((~invalid).all(-1)), dtype=dtype, order='F')


    @property
    def valid_coords(self) -> np.ndarray:
        """ Coordinate values for the valid locations """
        if self.is_sparse:
            if not self.uses_window:
                return self.coords
            coords = self.coords
            if hasattr(coords, 'todense'):
                n_dims = self._coords.shape[-1]
                coords = coords.data.reshape((-1, n_dims))
            return coords[self.valid_windows]
        return self.coords[tuple(self.valid_windows)]


    @property
    def valid_data(self) -> np.ndarray:
        """ Center data values for the valid locations """
        if self.is_sparse:
            # If no window is requested, the sparse data is already filtered
            if sum([np.sum(v) for v in self.window_depth.values()]+[0]) == 0:
                return self.sparse_data.data.reshape(-1, self.sparse_data.data.shape[-1])
            n_dims = self._data.shape[-1]
            data = self.sparse_data.data.reshape((-1, n_dims))
            return data[self.valid_windows]
        return self.data[tuple(self.valid_windows)]


    @property
    def valid_resolution(self) -> np.ndarray:
        """ Resolution for valid locations, parsing left/right if necessary """
        if self.is_uniform:
            return np.array(self.resolution)
        assert(not self.is_sparse), f'{self.resolution=}'
        try:
            res = [r[v] for r,v in zip(self.resolution, self.valid_windows)]
        except IndexError as e:
            raise Exception(
                'An index error here is likely due to blocks needing to be ' +
                're-cached (not the Dataset cache); this can be done by ' +
                'passing overwrite=True during Batcher creation.') from e
        return np.stack(res, axis=1)


    @cached_property
    def valid_mask(self) -> np.ndarray:
        """ Mask of valid pixels across all features """
        if self._valid_mask is not None:
            nbyte = f'{self._valid_mask.nbytes:,} bytes'
            shape = f'valid_mask {self._valid_mask.shape}'
            with self.benchmark(f'{shape} {nbyte}'):
                return self._valid_mask.compute()
        return ~self.invalid()


    @cached_property
    def fast_invalid_check(self) -> bool:
        """ Quick verification: returns True if no valid data exists """
        # If any dimension can have 0% existing and still be valid, don't check
        if not self.is_required:
            return False

        # TODO: this is_sparse block isn't necessary; valid_mask should(?) work
        if self.is_sparse:
            n = self.sparse_data.data.size
            if n and self.uses_window:
                return not self.valid_windows.any()
            return n == 0
        return not self.valid_mask.any()


    def summary(self, features: list[str] | None = None) -> xr.DataArray:
        """ Return summary statistics for data """
        features = self.feature_subset(features or self.features)
        data = self.dataset[features]

        quantiles = list(range(1, 100))#[2, 10, 25, 50, 75, 90, 98]
        perc_keys = [f'p{q}'.replace('p50', 'median') for q in quantiles]
        stat_keys = ['mean', 'std', 'min', 'max', 'null']
        all_stats = stat_keys + perc_keys

        if not features:
            return xr.DataArray(np.empty((0, len(all_stats))), coords={
                'features': [], 'statistics': all_stats}).chunk(-1)

        # Splitting large chunks seems to sometimes result in KeyError in dask
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            # Include in the summary stats any coordinates requested as features
            for coord in data.coords:
                if (coord in features) and (coord != 'datetime'):
                    coords = {c: data[c] for c in data.coords if c != coord}
                    data[f'{coord}_f'] = data[coord].astype('float32')
                    data[f'{coord}_f'] = data[[f'{coord}_f']] \
                        .expand_dims(**coords)[f'{coord}_f'] \
                        .transpose(*list(data.dims)) \
                        .chunk(data.chunksizes)

            # Slightly different handling required for sparse data
            if self.is_sparse:
                def calc(values):
                    # Explicitly apply functions per-block to avoid densifying
                    blocks = values.data.to_delayed().ravel()

                    # Ensure there's always at least one element to avoid errors
                    delays = [dask.delayed(np.append)(b.data, [np.nan]) for b in blocks]
                    series = [dask.delayed(pd.Series)(b) for b in delays]

                    # Fake array sizes so dask doesn't complain
                    length = [1] * len(series)
                    arrays = df.from_delayed(series).to_dask_array(lengths=length)
                    # Alternatively: values.data.map_blocks(lambda b: b.data[:,None,None])

                    # Similar to dense summary, but use dask operations directly
                    extra = dd(int, {'null' : lambda x: values.size-da.sum(da.isfinite(x))})
                    coord = xr.Variable('statistics', stat_keys)
                    value = lambda k: getattr(da, f'nan{k}', extra[k])(arrays).astype('float32')
                    toset = lambda k: xr.DataArray(value(k))
                    stats = xr.concat(map(toset, stat_keys), coord).to_dataset('statistics')
                    stats[perc_keys] = da.percentile(arrays, quantiles, internal_method='tdigest')
                    return stats.to_array('statistics')
                stats = data.apply(calc)

            # Dense summary computation relies mostly on xarray operations
            else:
                extra = dd(int, {'null' : lambda: data.isnull().sum(dtype='int64')})
                coord = xr.Variable('statistics', stat_keys)
                value = lambda k: getattr(data, k, extra[k])().to_array('features')
                stats = xr.concat(map(value, stat_keys), coord).to_dataset('statistics')

                # Compute percentiles as a group for efficiency
                stats[perc_keys] = xr.apply_ufunc(
                    lambda x: da.percentile(x.ravel(), quantiles,
                                            internal_method='tdigest'),
                    data, **{
                        'dask'             : 'allowed',
                        'input_core_dims'  : [list(data.dims)],
                        'output_core_dims' : [['statistics']],
                    }).to_array('features').to_dataset('statistics')
                stats = stats.to_array('statistics').to_dataset('features')

            stats = stats.rename({
                f'{c}_f':c for c in data.coords if f'{c}_f' in stats})
            stats = stats.to_array('features')
            stats['statistics'] = stats['statistics'].astype(str)
            stats['features'] = stats['features'].astype(str)
            return stats.chunk(-1)


    def set_valid_percents(self, valid_percent: dict):
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


    def set_feature_masks(self, feature_masks: dict):
        """ Mask the requested features so only values within a bin remain """
        # Need to modify valid_mask to handle zarr features
        assert(isinstance(self._data, list))

        def _get_feature(feature):
            if feature in self.features:
                index = self.features.index(feature)
                return self._data[index], index
            if feature in self.dims:
                index = self.dims.index(feature)
                coord = self.coords[..., index:index+1]
                assert(coord.shape == self._data[0].shape), (
                    f'Unexpected coord shape: {self.coords.shape=} vs '
                    f'data.shape={self._data[0].shape}')
                return coord, 0
            raise Exception(f'Unknown {feature=}: {self.features=}')

        self._feature_mask = True
        for feature in self.feature_subset(list(feature_masks)):
            label = feature.split('@')[0]
            lo,hi = feature_masks[feature]
            val,i = _get_feature(label)
            valid = (val >= lo) & (val < hi)
            self._valid_mask &= valid
            # self._data[i] = da.where(valid, self._data[i], np.nan)
            self._feature_mask &= valid

        if isinstance(self._feature_mask, bool):
            self.__dict__.pop('_feature_mask')
        self.__dict__.pop('valid_mask', None)


    def reset_valid_percents(self):
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


    def extract(self,
        matches : np.ndarray,
        return_features : list[str] | None = None,
        empty : bool = False,
    ):# -> dict[Int, xr.Dataset]:
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
            orig_dims = ['features'] + [d for d in orig_dims if d!='features']
            keep_dims = [f for f in self.dims if f in orig_dims] + ['features']

            # Extract only the requested features to be returned
            available = np.array(list(features) + list(orig_dims)[1:])
            requested = [f.split('@')[0] for f in return_features]

            # Cast dtypes for each feature later (to avoid mixed type array)
            mask = np.isin(available, requested)
            keys = dict(zip(requested, return_features))
            dtypes = {keys[k]: dtypes[k] for k in available[mask]}

            # Create arrays for the left and total window depths
            lower = np.array([self.window_depth[k][0] for k in self.dims])
            total = np.array([self.window_total[k]    for k in self.dims])
            shape = tuple(s for s,d in zip(total, self.dims) if d in orig_dims)
            ndims = len(self.dims)

            # If this was a dropped Block, return just a NaN array placeholder
            if empty:
                return (
                    np.nan * np.zeros((len(requested),) + shape, 'float32'),
                    dtypes,
                )

            # Otherwise, get all unique match indices
            unique, indices = np.unique(matches, return_inverse=True)

            # Calculate the lower and upper bounds for each window dimension
            center = np.array(self.valid_windows)[:, unique, None]
            center-= lower[:, None, None]
            bounds = [left + np.arange(size)
                      for left, size in zip(center, total)]

            # Remove virtual dimension from coordinate features
            real_dims = sorted(map(self.dims.index, orig_dims[1:]))
            n_samples = len(unique)

            def expand(axis: int, bound: np.ndarray) -> np.ndarray:
                """ Add axes to each bound based on the dim it applies to """
                axes = list(set(range(1 ,ndims+1)) - {axis})
                return np.expand_dims(bound, axes)

            # Expand the bounds so they can be broadcast over the full data/coords
            windows = tuple(starmap(expand, enumerate(bounds, 1)))
            assert(len(windows[0]) == n_samples), [len(windows), n_samples]

            # Extract windows and drop virtual dimensions
            # Note: SegFault/Access violation here is likely an issue with data
            #       that is cached on disk; try removing the cache to resolve.
            data = np.append(self.data, self.coords[..., real_dims], axis=-1)
            data = data[..., mask][windows].reshape(
                (n_samples,) + shape + (len(requested),))

            # Transpose data to the correct order: [samples, features, ...]
            data = data.transpose(tuple(
                map((['']+keep_dims).index, ['']+orig_dims)
            ))
            return data[indices], dtypes

        orig_dims, features, dtypes, req_features = self.original_dims
        orig_dims = ['features'] + [d for d in orig_dims if d != 'features']

        # Create arrays for the left and total window depths
        lower = np.array([self.window_depth[k][0] for k in self.dims])
        total = np.array([self.window_total[k]    for k in self.dims])
        ndims = len(self.dims)
        shape = tuple(s for s,d in zip(total, self.dims) if d in orig_dims)

        # Initialize the final result dict with all globally applicable values
        keep_dims = [f for f in self.dims if f in orig_dims] + ['features']
        xr_kwargs = {
            'dims' : orig_dims,
            'key_label' : self.key_label,
            'requested_features' : req_features,
        } | ({'attrs' : {'resolution': dict(zip(self.dims, self.resolution))}}
                if self.is_uniform else {})

        def cast_dtype(key: str, value: np.ndarray) -> np.ndarray:
            """ Cast the given value array back to its original dtype """
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
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
                     'coords' : dict(zip(keep_dims,
                                    map(cast_dtype, keep_dims, coords))) | {
                    'features': features} } | dict(xr_kwargs)

        nan_matches = ~np.isfinite(matches)

        # If this was a dropped Block, use just NaN array placeholders
        if empty or nan_matches.all():
            data = np.nan * np.zeros((len(features),)+shape, 'float32')
            coords = [np.nan * np.zeros(s, 'float32') for s in shape]
            values = [gen_xr_dict(data, *coords)]
            if empty:
                return values
            return [values * len(np.atleast_1d(idx)) for idx in matches]

        # Replace any NaN matches with -1 to select last item
        matches[nan_matches] = -1
        matches = matches.astype(int)

        # Get all unique match indices and drop any -1 values
        ravel   = lambda v: getattr(v, 'ravel', lambda: v)()
        indices = tuple(np.unique([j for i in matches
                                   for j in ravel(i) if j != -1]).astype(int))
        samples = len(indices)

        # Calculate the lower and upper bounds for each window dimension
        if not self.is_sparse:
            center = np.array(self.valid_windows)[:, indices, None]
            center-= lower[:, None, None].astype(center.dtype)
            bounds = [left + np.arange(size, dtype=center.dtype)
                      for left, size in zip(center, total)]
            # Expand bounds so they can be broadcast over the full data/coords
            windows = tuple(starmap(expand, enumerate(bounds, 1)))
            assert(len(windows[0]) == samples), [len(windows), samples]

        # Extract windows and drop virtual dimensions
        # Note: SegFault/Access violation here is likely an issue with data
        #       that is cached on disk; try removing the cache to resolve.
        # data = self.data[windows].reshape((samples,)+shape+(len(features),))
        is_arr = isinstance(self._data, da.Array)
        n_feat = self._data.shape[-1] if is_arr else len(self._data)
        get_ix = lambda i: self._data[..., i] if is_arr else self._data[i]

        # Sparse data matches require different handling than dense arrays
        if self.is_sparse:
            indices = list(indices)

            # If a window is requested along at least one dimension, we need
            # to construct a flattened grid of indices to pull from the 'dense'
            # COO object representation (as advanced indexing isn't available)
            if self.uses_window:

                # Gather window centers, add dimensions to allow broadcasting
                newdim = tuple(range(ndims))
                center = self.sparse_data[..., 0].coords[:, self.valid_windows]
                window = np.expand_dims(center.T[indices] - lower, newdim).T

                # Create the flattened meshgrid for indexing into sparse.COO
                offset = map(np.arange, total)
                offset = np.meshgrid(indices, *offset, indexing='ij')[1:]
                window = [dim+right for dim, right in zip(window, offset)]
                window = tuple(w.ravel() for w in window)

                # Extract flat windows from the coordinate vectors and reshape
                shapes = [samples] + list(total)
                c_vecs = self.coord_vecs
                coords = [v[w].compute() for v, w in zip(c_vecs, window)]
                coords = np.stack(coords, axis=-1).reshape(shapes + [ndims])

                # Extract flat windows from the data, reshape, and materialize
                data = self.sparse_data[window]
                data = data.reshape(shapes + [len(features)]).todense()

                # Remove any virtual dimensions
                idxs = [slice(None)] * data.ndim
                for i, d in enumerate(self.dims, 1):
                    if d not in keep_dims:
                        idxs[i] = 0
                data = data[tuple(idxs)]

            # If no window is requested, we can just pull directly out of the
            # sparse representation rather than requiring dense materialization
            else:
                shape = (samples,) + shape
                if self.sparse_data.data.size:
                    data = np.stack([self.sparse_data[..., i].data[indices]
                                     for i in range(len(features))], axis=-1)
                    data = data.reshape(shape+(len(features),))
                    coords = self.coords[indices].reshape(
                                        shape+(len(self.dims),))
                else:
                    data = np.nan*np.zeros(shape+(len(features),),'float32')
                    coords = np.nan*np.zeros(shape+(len(self.dims),),'float32')

        else:
            shape  = (samples,) + shape
            delays = [dask.delayed(get_ix(i))[windows] for i in range(n_feat)]
            data   = da.compute(*delays)
            data   =   np.stack(data, -1).reshape(shape+(len(features),))
            coords = self.coords[windows].reshape(shape+(len(self.dims),))

        for k in ['coords', 'valid_windows', 'valid_mask', 'sparse_data'][:-1]:
            self.__dict__.pop(k, None)

        # Remove virtual dimension from coordinate features
        coords = coords[..., sorted(map(self.dims.index, orig_dims[1:]))]

        # Transpose data to the correct order: [samples, features, ...]
        data = data.transpose(tuple(
            map((['']+keep_dims).index, ['']+orig_dims)
        ))

        # Extract coordinate vectors
        coord_vectors = starmap(collapse, enumerate(coords.T))
        coord_vectors = list(map(cast_dtype, keep_dims, coord_vectors))

        # Extract final window dictionaries to use for Sample initialization
        windows = dict(zip(indices, map(gen_xr_dict, data, *coord_vectors)))
        if nan_matches.any():
            val = windows[indices[0]]
            to_nan = lambda v: np.ones_like(v, dtype='float32') * np.nan
            windows[-1] = val | {
                'data'   : [v*np.nan for v in val['data']],
                'coords' : {k: (v if k == 'features' else
                    to_nan(v).astype(v.dtype))
                    for k, v in val['coords'].items()
                },
            }
        return [[windows[i] for i in np.atleast_1d(idx)] for idx in matches]


    def feature_subset(self, features: Sequence|None = []) -> list[str] | None:
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
            If an empty list is given, all available features are returned;
            if None is given, None is returned.

        Returns
        -------
        list[str]
            The subset of features that this Block can provide, returned as
            a (flat) list of strings. Note that this list includes coordinate
            dimension names as well as data feature names.

        """

        # Features being None means Sample objects will be created, so ignore
        if features is not None:
            available_features = chain(*map(list, self.original_dims[:2]))
            available_features = list(set(available_features)-{'features'})

            def recurse(feature: Sequence | str) -> list:
                """ Recurse over the nested structure to find availablity """

                # Feature is still nested if it's not a string
                if not isinstance(feature, str):
                    return chain(*map(recurse, feature or available_features))

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
