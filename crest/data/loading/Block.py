from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from collections import defaultdict as dd
from functools import cached_property, reduce
from dask.delayed import Delayed
from itertools import chain
from numbers import Number, Integral as Int
from copy import copy as shallow_copy

import dask.array as da
import bottleneck as bn
import xarray as xr
import numpy as np
import warnings
import dask

from crest.base import BaseAbstract
from crest.utils import Stopwatch


class Block(BaseAbstract):
    """Wrap a dense Dask block and its sampling configuration.

    Notes
    -----
    Datafile supplies separate feature arrays, a prepared validity mask, and
    rectilinear coordinate vectors. The constructor defaults remain available
    for metadata-only use; data access relies on these prepared inputs.
    
    Parameters
    ----------
    data          : Collection[da.Array]
        Separate Dask feature arrays supplied by Datafile, each retaining
        its singleton final feature axis.
    coords        : da.Array
        Dask Array object containing the coordinates which define the block.
    inbound_mask  : da.Array
        Dask Array object indicating which elements are inside the bounds
        of the data (i.e. not padding). Note virtual dims are always in-bound.
    overlap_mask  : da.Array
        Dask Array object indicating which elements are part of an overlapped
        chunk (True), vs part of the original center chunk (False).
    valid_mask    : da.Array
        Prepared validity mask supplied by Datafile.
    dims          : Collection[str]
        Dimension names, ordered the same as the `data` axes.
    original_dims : Collection
        Contains original dimension ordering as well as the features coordinate
        vector. Todo: modify this to pass the information in more cleanly.
    resolution    : Collection[Number]
        Resolution for each dimension, ordered the same as the `data` axes.
    window_depth  : dict[str, Collection[Int]]
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
    allow_repeats : bool
        If True, allow block overlap regions to be considered when generating
        valid window centers. With this as True, it means that all possible
        samples should be generated, but there is the potential for duplicate
        samples to be generated across different blocks at the boundaries. If
        instead this is False (the default), there will not be any duplicated
        samples generated, but instead there *may* be samples missing at the
        block boundaries.

    """
    def __init__(self,
        data          : Collection[da.Array],
        coords        : da.Array,
        inbound_mask  : da.Array,
        overlap_mask  : da.Array,
        valid_mask    : da.Array,
        resolution    : da.Array | Collection       = [],
        dims          : Collection[str]             = [],
        original_dims : Sequence                    = [],
        window_depth  : dict[str, Collection[Int]]  = {},
        valid_percent : dict[tuple[str], Number]    = {},
        allow_repeats : bool  = False,
        block_index   : int   = 0,
        label         : str   = '',
        sparsity      : Number | da.Array = 0.,
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
        self.allow_repeats = allow_repeats
        self.block_index   = block_index
        self.coord_vecs = coord_vecs
        self.key_label = key_label

        # Shape sanity checks
        arrays = [coords, inbound_mask, overlap_mask]
        shapes = [v.shape[:-1] for v in arrays]
        assert(len(set(shapes)) == 1), shapes
        assert(set(map(len, shapes)) == {len(dims)}), [shapes, dims]
        
        vector_shape = tuple(vector.shape[0] for vector in coord_vecs)
        assert vector_shape == coords.shape[:-1], (
            f'{self}: {vector_shape=} != {coords.shape[:-1]=}')
        assert len(self.features) == len(self._data), (
            f'{self}: {len(self.features)=} != {len(self._data)=}; '
            f'{self.features=}')

    
    def __repr__(self) -> str:
        return f'{type(self).__name__}[{self.label}]'


    def copy(self) -> Block:
        """ Return a shallow copy of all object attributes """
        block = shallow_copy(self)
        block.__dict__.pop('benchmark', None)
        return block


    @cached_property
    def benchmark(self) -> Callable:
        """ Helper for benchmarking code timing and memory usage """
        logger  = print if not hasattr(self, 'logger') else self.logger.debug
        message = f'\t\t\t{self}'
        silent  = not hasattr(self, 'logger')
        return lambda label, **kwargs: Stopwatch(**({
            'message' : f'{message}.{label}',
            'logger'  : logger,
            'silent'  : silent,
        } | kwargs))


    @property
    def is_required(self) -> bool:
        """ Whether this block is required to form a valid sample """
        return np.prod(list(self.valid_percent.values())) > 0


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
        """ Compute and concatenate the supplied feature arrays """
        with self.benchmark('data'):
            return np.concatenate(da.compute(*self._data), axis=-1)


    @property
    def dataset(self) -> xr.Dataset:
        """ Build a lazy xarray view """
        arrays = self._data
        values = [(self.dims, array[..., 0]) for array in arrays]
        return xr.Dataset(**{
            'data_vars' : dict(zip(self.features, values)),
            'coords'    : dict(zip(self.dims, self.coord_vecs)),
        })


    @cached_property
    def coords(self) -> np.ndarray:
        """Compute the coordinate grid only on first use."""
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
        compute = lambda v: v.compute() if hasattr(v, 'compute') else v
        return list(map(compute, self._resolution))


    @cached_property
    def dtype(self) -> np.dtype:
        """ Create a composite datatype based on shapes of the data windows """
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
    def window_depth(self) -> dict[str, Collection[Int]]:
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


    @cached_property
    def valid_windows(self) -> np.ndarray:
        """ Determine indices for all valid sample windows in the block """

        def moving_sum(invalid: np.ndarray, keys_pct: tuple) -> np.ndarray:
            """ Fast moving sum using bottleneck """
            if invalid.all(): 
                return invalid
            keys, percent = keys_pct

            # Valid percent is handled in a way that allows a small percent
            # to be used in indicating 'at least one element'. For example,
            # in a 3x3 spatial window a valid percent of 0.01 could be used
            # to ensure at least one element in windows are valid. In other
            # words, percentages are interpreted as ceil(n_elements * pct).
            n_total = int(np.prod([self.window_total[k] for k in keys]))
            require = int(np.ceil(np.nextafter(percent*n_total, 0.)))
            maximum = n_total - min(n_total, require)

            # Multiple axes can be used to define a combined valid percent
            # e.g. {(ax1, ax2): 0.8} -> combination of the two need 80% valid
            pad_val = 1
            for key in keys:
                axis = self.axes[key]
                size = self.window_total[key]

                if size > invalid.shape[axis]:
                    raise ValueError(f'{self}: Block dimension "{key}" with ' +
                        f'size={invalid.shape[axis]} is too small for ' +
                        f'requested window size={size}')

                # Width-one rolling sums are an identity operation
                if size == 1: continue

                # Calculate window offsets and padding
                offsets = (slice(None),) * axis + (slice(size - 1, None),)
                padding = [(0,0)] * invalid.ndim
                padding[axis] = self.window_depth.get(key, (0,0))

                # Pad the invalid mask in order to center the window
                invalid = invalid.astype('int32') # Bottleneck is faster w/ int
                invalid = np.pad(invalid, padding, constant_values=pad_val)

                # Offset the new invalid elements so the window is centered
                invalid = bn.move.move_sum(invalid, size, axis=axis)[offsets]
                pad_val*= size
            return invalid > maximum

        # If some elements need to be valid, we need to check for valid windows
        if self.is_required:
            # Datafile has already collapsed feature validity into one mask
            invalid = ~self.valid_mask[..., 0]

            # Calculate the running invalid mask over all dimensions
            invalid = reduce(moving_sum, self.valid_percent.items(), invalid)

        # Otherwise, checking is unnecessary since 0% are required to be valid
        else: invalid = np.zeros(self.shape, dtype='bool')

        # Mask overlapped elements, as they cannot be window centers
        if not self.allow_repeats and self.block_index == 0:
            invalid |= self.overlap_mask[..., 0]

        # Mask elements outside the bounds of the data (except virtual)
        invalid |= ~self.inbound_mask[..., 0]

        # Bin masks restrict centers as well as contributing to window validity
        if hasattr(self, '_feature_mask'):
            invalid |= ~self._feature_mask.compute()[..., 0]

        # Clip only dimensions in the policy
        for keys in self.valid_percent:

            for dim in keys:
                offset = (slice(None),) * self.axes[dim]
                lo, hi = self.window_depth[dim]
                hi = invalid.shape[self.axes[dim]] - hi
                invalid[offset + (slice(None, lo),)] = True
                invalid[offset + (slice(hi, None),)] = True

        # Store integer centers without carrying the singleton feature axis
        largest = max(self.shape, default=0) - 1
        dtype = np.min_scalar_type(max(0, largest))
        if dtype.itemsize < 2: dtype = np.dtype('uint16')
        return np.array(np.where(~invalid), dtype=dtype, order='F')


    @property
    def valid_coords(self) -> np.ndarray:
        """ Valid coordinates without materializing the full grid """
        indices = self.valid_windows
        if not indices.shape[1]:
            return np.empty((0, self.ndim), dtype=self._coords.dtype)

        # A single-chunk lookup must not split repeated indices into tasks
        vectors = []
        for vector, index in zip(self.coord_vecs, indices):
            if isinstance(vector, da.Array) and vector.numblocks == (1,):
                vector = vector.to_delayed(optimize_graph=False).item()
            vectors.append(vector[index])
        return np.stack(dask.compute(*vectors), axis=-1)


    @property
    def valid_resolution(self) -> np.ndarray:
        """ Resolution for valid locations, parsing left/right if necessary """
        if self.is_uniform:
            return np.array(self.resolution)
        try:
            res = [r[v] for r,v in zip(self.resolution, self.valid_windows)]
        except IndexError as e:
            raise IndexError(
                'An index error here is likely due to blocks needing to be ' +
                're-cached (not the Dataset cache); this can be done by ' +
                'passing overwrite=True during Batcher creation.') from e
        return np.stack(res, axis=1)


    @cached_property
    def valid_mask(self) -> np.ndarray:
        """ Compute the element-wise validity mask """
        nbyte = f'{self._valid_mask.nbytes:,} bytes'
        shape = f'valid_mask {self._valid_mask.shape}'
        with self.benchmark(f'{shape} {nbyte}'):
            return self._valid_mask.compute()


    @cached_property
    def fast_invalid_check(self) -> bool:
        """ Return True when a required block has no valid data """
        if not self.is_required:
            return False
        return not self.valid_mask.any()


    def summary(self, features: list[str] | None = None) -> xr.DataArray:
        """ Return summary statistics for data """
        features = self.feature_subset(features or self.features, parse=True)
        data = self.dataset[features]

        quantiles = list(range(1, 100))
        perc_keys = [f'p{q}'.replace('p50', 'median') for q in quantiles]
        stat_keys = ['mean', 'std', 'min', 'max', 'null']
        all_stats = stat_keys + perc_keys

        if not features:
            return xr.DataArray(np.empty((0, len(all_stats))), coords={
                'features': [], 'statistics': all_stats}).chunk(-1)

        # Splitting large chunks seems to sometimes result in KeyError in dask
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            # Include in the summary stat any coordinates requested as features
            for coord in data.coords:
                if (coord in features) and (coord != 'datetime'):
                    coords = {c: data[c] for c in data.coords if c != coord}
                    data[f'{coord}_f'] = data[coord].astype('float32')
                    data[f'{coord}_f'] = data[[f'{coord}_f']] \
                        .expand_dims(**coords)[f'{coord}_f'] \
                        .transpose(*list(data.dims)) \
                        .chunk(data.chunksizes)

            stats = self._summary(data, stat_keys, perc_keys, quantiles)

            stats = stats.rename({
                f'{c}_f':c for c in data.coords if f'{c}_f' in stats})
            stats = stats.to_array('features')
            stats['statistics'] = stats['statistics'].astype(str)
            stats['features'] = stats['features'].astype(str)
            return stats.chunk(-1)


    def _summary(self,
        data      : xr.Dataset,
        stat_keys : list[str],
        perc_keys : list[str],
        quantiles : list[int],
    ) -> xr.Dataset:
        """ Calculate statistics using the dense xarray representation """
        extra = dd(int, {
            'null' : lambda: data.isnull().sum(dtype='int64'),
        })
        coord = xr.Variable('statistics', stat_keys)

        get_f = lambda k: getattr(data, k, extra[k])()
        value = lambda k: get_f(k).to_array('features')
        array = xr.concat(map(value, stat_keys), coord)
        stats = array.to_dataset('statistics')

        # Compute percentiles as a group for efficiency
        stats[perc_keys] = xr.apply_ufunc(
            lambda x: da.percentile(x.ravel(), quantiles, **{
                'internal_method' : 'tdigest',
            }),
            data, **{
                'dask'             : 'allowed',
                'input_core_dims'  : [list(data.dims)],
                'output_core_dims' : [['statistics']],
            }).to_array('features').to_dataset('statistics')
        stats = stats.to_array('statistics').to_dataset('features')
        return stats


    def _get_coordinate(self, index: int) -> da.Array:
        """ Return coordinate values for one axis """
        return self._coords[..., index:index+1]


    def set_feature_masks(self, feature_masks: dict) -> None:
        """ Mask the requested features so only values within a bin remain """
        # Sampling filters change masks, not the source feature values
        assert(isinstance(self._data, list))

        def _get_feature(feature: str) -> tuple:
            if feature in self.features:
                index = self.features.index(feature)
                return self._data[index], index
            if feature in self.dims:
                index = self.dims.index(feature)
                coord = self._get_coordinate(index)
                assert(coord.shape == self._data[0].shape), (
                    f'Unexpected coord shape: {self.coords.shape=} vs '
                    f'data.shape={self._data[0].shape}')
                return coord, 0
            raise ValueError(f'Unknown {feature=}: {self.features=}')

        mask = True
        features = self.feature_subset(list(feature_masks))
        for feature in features:
            label = self._parse_feature(feature)
            lo,hi = feature_masks[feature]
            val,i = _get_feature(label)
            valid = (val >= lo) & (val < hi)
            mask &= valid

        # Retain separate window-content and center restrictions
        if isinstance(mask, bool):
            self.__dict__.pop('_feature_mask', None)
        else:
            # Note no in-place mutation occurs, as _valid_mask is a dask array
            self._valid_mask &= mask
            self._feature_mask = mask

        # Reset any cached properties that sampling masks affect
        self._clear_validity()


    def set_valid_percents(self, valid_percent: dict) -> None:
        """ Set a new valid_percent after formatting correctly """
        assert(getattr(self, '_original_valid_percent', None) is None)
        self._original_valid_percent = self.valid_percent
        formatted = {}
        keys = ()
        for k, v in valid_percent.items():
            if not isinstance(k, tuple):
                k = (k,)
            keys += k
            formatted[k] = v
        for d in self.dims:
            if d not in keys:
                formatted[(d,)] = 1
        self.valid_percent = formatted
        self._clear_validity()


    def _clear_validity(self) -> None:
        """ Discard cached attributes after a sampling-policy change """
        for key in ['valid_mask', 'valid_windows', 'fast_invalid_check']:
            self.__dict__.pop(key, None)


    def _parse_feature(self, feature: str) -> str:
        """ Discard any selectors to return only the actual feature name """
        return f'__{feature}'.split('@')[0].split('__', 2)[-1]

    
    def cleanup(self) -> None:
        """ Release computed objects but retain lazy source graphs """
        for key in ['valid_windows', 'valid_mask', 'coords',
                    'fast_invalid_check']:
            self.__dict__.pop(key, None)


    def _gather_windows(self,
        windows     : tuple,
        features    : list[int],
        coordinates : list[int],
    ) -> tuple:
        """Gather requested values in one Dask computation.

        Parameters
        ----------
        windows : tuple
            Broadcastable integer indices, with samples on the first axis.
        features : list[int]
            Data-feature positions to read, in output order.
        coordinates : list[int]
            Coordinate-component positions to read, in output order.

        Returns
        -------
        tuple
            Data and optional coordinate arrays, with components on the last
            axis. Only the selected windows are materialized as output arrays.
        """
        arrays = [self._data[i][..., 0] for i in features]

        def select(
            array: da.Array | np.ndarray,
        ) -> da.Array | np.ndarray | Delayed:
            """ Preserve shared graph keys and prune unused input chunks """
            if not isinstance(array, da.Array): return array[windows]
            if all(count == 1 for count in array.numblocks):
                return array.to_delayed(optimize_graph=False).item()[windows]
            return array.vindex[windows]

        # Joint computation must see original shared source keys. Converting
        # each feature with delayed(array) can optimize the readers separately.
        delays = [select(array) for array in arrays]

        # Select grid components first, preserving padding and shared reads
        if coordinates:
            coords = self.__dict__.get('coords', self._coords)
            delays.extend(select(coords[..., i]) for i in coordinates)
        values = dask.compute(*delays)

        shape = np.broadcast_shapes(*(index.shape for index in windows))
        data  = (np.stack(values[:len(features)], axis=-1) if features else
                 np.empty(shape + (0,), dtype=self._coords.dtype))
        coord = (np.stack(values[len(features):], axis=-1)
                 if coordinates else None)
        return data, coord


    def extract(self,
        matches         : np.ndarray,
        return_features : list[str] | None = None,
        empty           : bool = False,
    ) -> list | tuple:
        """Extract windows as raw sample dictionaries or direct feature arrays.
        
        Notes
        -----
        Neither output path constructs an xarray object. Subclasses override
        only the storage-specific gather and valid-center representation.
        
        Parameters
        ----------
        matches : np.ndarray
            Positions within valid centers. Repeated positions reuse the same
            extraction. Non-finite entries denote missing optional matches.
        return_features : list[str] | None
            Features already selected for this block by feature_subset. None
            requests raw dictionaries for Sample initialization; otherwise
            return an array and its ordered feature-dtype mapping.
        empty : bool
            Return one missing-block template without reading any source data.

        Returns
        -------
        list | tuple
            Nested window dictionaries, or (values, dtypes). Direct arrays have
            match axes first, followed by features and original spatial axes.
            Empty-block templates omit the match axes for Blockset to expand.

        """
        original, features, dtypes = self.original_dims
        real_dims = [dim for dim in original if dim != 'features']
        keep_dims = [dim for dim in self.dims if dim in real_dims]
        out_dims  = ['features'] + real_dims
        depths    = self.window_depth
        totals    = self.window_total
        shape     = tuple(totals[dim] for dim in real_dims)
        raw       = return_features is None

        # Resolve metadata before reading anything, including missing templates
        if raw:
            positions  = list(range(len(features)))
            coord_keys = real_dims
            metadata = {
                'dims'               : out_dims,
                'key_label'          : self.key_label,
            }
            if self.is_uniform:
                resolution = dict(zip(self.dims, self.resolution))
                metadata['attrs'] = {'resolution': resolution}
        else:
            labels = {self._parse_feature(key): key for key in return_features}
            positions  = [i for i, key in enumerate(features) if key in labels]
            data_keys  = [features[i] for i in positions]
            coord_keys = [key for key in real_dims if key in labels]
            types = {labels[k]: dtypes[k] for k in data_keys + coord_keys}

        def cast(key: str, value: np.ndarray) -> np.ndarray:
            """ Restore source dtypes only for raw Sample output """
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                return value.astype(dtypes[key], copy=False)

        def missing_window() -> dict:
            """ Build a metadata-only placeholder for an unmatched source """
            values = [cast(key, np.full(shape, np.nan, dtype='float32'))
                      for key in features]
            coords = {dim: cast(dim, np.full(totals[dim], np.nan, 'float32'))
                      for dim in real_dims}
            return {
                'data'   : values,
                'coords' : coords | {'features': features},
            } | metadata

        missing = ~np.isfinite(matches) | (matches == -1)
        if empty or missing.all():
            if raw:
                window = missing_window()
                if empty: return [window]
                return [[window] * len(np.atleast_1d(index))
                        for index in matches]
            dims = (len(types),) + shape
            if not empty: dims = matches.shape + dims
            return np.full(dims, np.nan, dtype='float32'), types

        # Index a separate array; never replace NaNs in the caller's matches
        indices = np.where(missing, -1, matches).astype(np.intp)
        unique  = np.unique(indices[~missing])
        centers = self.valid_windows[:, unique].astype(np.intp, copy=False)
        windows = []
        for axis, dim in enumerate(self.dims):
            bound = centers[axis, :, None] - depths[dim][0]
            bound = bound + np.arange(totals[dim])
            sizes = [len(unique)] + [1] * self.ndim
            sizes[axis + 1] = totals[dim]
            windows.append(bound.reshape(sizes))

        # Subclasses implement the entire gather, not per-element dispatch
        data, coords = self._gather_windows(tuple(windows), **{
            'features'    : positions,
            'coordinates' : [self.dims.index(key) for key in coord_keys],
        })

        # Virtual dimensions have unit window depth and are omitted from output
        sizes = (len(unique),) + tuple(totals[dim] for dim in keep_dims)
        axes  = [0, len(keep_dims) + 1]
        axes += [keep_dims.index(dim) + 1 for dim in real_dims]

        if not raw:
            values = (data if coords is None else
                      np.concatenate((data, coords), axis=-1))
            values = values.reshape(sizes + (len(types),)).transpose(axes)
            mapped = np.searchsorted(unique, np.maximum(indices, 0))
            mapped = np.minimum(mapped, len(unique) - 1)
            output = values[mapped]
            if missing.any():
                if not np.issubdtype(output.dtype, np.inexact):
                    output = output.astype('float64')
                output[missing] = np.nan
            return output, types

        # Raw windows retain original coordinate-vector order and dtype
        data = data.reshape(sizes + (len(features),)).transpose(axes)
        if coords is not None:
            coords = coords.reshape(sizes + (len(coord_keys),))
        vectors = {}
        for component, dim in enumerate(coord_keys):
            selector = [slice(None)] + [0] * len(keep_dims) + [component]
            selector[keep_dims.index(dim) + 1] = slice(None)
            vectors[dim] = cast(dim, coords[tuple(selector)])

        result = {}
        for row, index in enumerate(unique):
            result[index] = {
                'data'   : [cast(key, values) for key, values in
                            zip(features, data[row])],
                'coords' : {dim: vec[row] for dim, vec in vectors.items()}
                           | {'features': features},
            } | metadata
            
        if missing.any():
            result[-1] = missing_window()
            values = result[unique[0]]['data']
            result[-1]['data'] = [value * np.nan for value in values]

        for key in ['coords', 'valid_windows', 'valid_mask']:
            self.__dict__.pop(key, None)
        return [[result[index] for index in np.atleast_1d(group)]
                for group in indices]


    def feature_subset(self, 
        features : Sequence | None = [], 
        parse    : bool = False,
    ) -> list[str] | None:
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
        parse : bool
            Whether the returned feature strings should be parsed into their
            actual feature names (e.g. 'feature@1' returned as 'feature').

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

                # Also need to handle __{source} syntax, to allow feature
                # selection by Datafile label (rather than solely its index)
                *extra, source, name = f'__{feature}'.split('__', 2)

                if source and source.replace('$', '/') == self.key_label:
                    if name in available_features:
                        return [name if parse else feature]
                    
                    # Duplicate key_labels are possible if a Datafile requests
                    # multiple window sizes across its features
                    return []
                
                # Also need to handle @{index} syntax, to allow feature
                # selection by Datafile when there are duplicated features
                name, i, *extra = (feature+f'@{self.block_index}').split('@')

                try: 
                    i = int(i)
                except:
                    raise ValueError(f'"@" selector needs an int: {feature=}')
                    
                if i == self.block_index:
                    if name in available_features:
                        return [name if parse else feature]

                    # If any indices remain, it means this block was explicitly
                    # requested to provide this feature - which it doesn't have
                    assert(len(extra)==0), f'{feature} unavailable from {self}'
                return []
            return list(set( recurse(features or available_features) ))
