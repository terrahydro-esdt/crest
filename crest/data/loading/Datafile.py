from __future__ import annotations

from collections.abc import Collection, Callable
from collections import defaultdict as dd
from fsspec.mapping import FSMap
from functools import cached_property, partial
from itertools import starmap, product
from numbers import Number, Integral as Int
from pathlib import Path
from typing import ClassVar
from tlz import dissoc, valmap

import dask.dataframe as df
import dask.array as da
import xarray as xr
import pandas as pd
import numpy as np
import warnings
import hashlib
import shutil
import ctypes
import math
import dask
import copy
import sys

from crest.base import BaseAbstract
from crest.utils import S3Path, Stopwatch, dask_overlap
from crest.utils.logging_service import get_logger
from .backend import get_backend
from .Block import Block


# Bool type which allows numpy bools as well
Bool = bool | np.bool_


class Datafile(BaseAbstract):
    """Class which handles loading data from a single source.

    Parameters
    ----------
    location      : Path | str | FSMap | S3Path | xr.Dataset,
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
    match_radius  : dict[str, str | Number]
        Allows specification of the radius around each point that will match
        coordinates from other Datafiles. This is a dictionary in the form of
        {'dimension_name': radius}, where radius can be a string that indicates
        a percentage of the dimension resolution that should be used (e.g.
        data with a datetime dimension that has daily resolution could use
        {'datetime':'25%'} to indicate only points within +/- 6 hours should
        match); or radius can be a number that will be used as the exact radius
        around each point (e.g. with the same data as before, {'datetime': 120}
        would allow matches of +/- 2 hours around each point). Note that the
        units of datetime dimensions will always be minutes, and so the radius
        value should be specified in minutes if using an exact value.
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
    allow_repeats : bool
        If True, allow block overlap regions to be considered when generating
        valid window centers. With this as True, it means that all possible
        samples should be generated, but there is the potential for duplicate
        samples to be generated across different blocks at the boundaries. If
        instead this is False (the default), there will not be any duplicated
        samples generated, but instead there *may* be samples missing at the
        block boundaries.
    is_required : bool
        To globally distribute coordinates ~uniformly across blocks within a
        Dataset, some Datafiles (e.g. sparse data) need to be allowed to have
        blocks with zero samples - thus allowing other Datafiles' blocks to be
        properly aligned while maintaining ~equal coordinate distribution. The
        `is_required` parameter indicates whether this Datafile is required to
        be present within a Sample, for that Sample to be considered valid. In
        other words, when `is_required=False`, a valid Sample can be generated
        even if this Datafile is missing data at the Sample's location (e.g.
        this Datafile may be one of several that contain targets, where only
        one target must be present in order to generate a valid sample).
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
    DEFAULT_WINDOW_SIZE: ClassVar[int] = 0

    # Full window must be valid for undefined dimensions
    DEFAULT_VALID_PERCENT: ClassVar[int] = 1

    # Ensure we include int(32/64) representations of NaN
    DEFAULT_INVALID_VALUES: ClassVar[list[Number]] = [
        -2147483648, -9223372036854775808, float('inf'), -float('inf'),
    ]

    # Any attributes set in __init__ which are not contained in this list will
    # be included in the hash calculation that is used to name the cache folder
    CACHE_KEY_EXCLUSIONS: ClassVar[str] = [
        '_window_depth', 'match_radius', 'dataset_index', 'is_required',
    ]


    def __init__(self,
        location      : Path | str | FSMap | S3Path | xr.Dataset,
        features      : list[str] = [],
        extent        : dict[str, Collection] = {},
        window_depth  : dict[str, int | Collection[int] | None] = {},
        match_radius  : dict[str, str | Number] = {},
        valid_percent : dict[str | tuple[str], Number] = {},
        invalid_value : object = [],
        preprocessors : list[Callable] = [],
        sort_dims     : bool = True,
        allow_repeats : bool = False,
        no_overlaps   : bool = False,
        region        : list[str] | None = None,
        key_label     : str | None = None,
        dataset_index : int = 0,
        is_required   : bool = True,
        **kwargs
    ):
        if isinstance(location, FSMap):
            location = S3Path(location)

        self.location = location
        self.features = sorted(features)
        self.extent   = extent.copy()
        self._kwargs  = kwargs
        self._window_depth  = window_depth.copy()
        self._valid_percent = valid_percent.copy()
        self._invalid_value = invalid_value
        self.match_radius   = match_radius
        self.preprocessors  = preprocessors
        self.sort_dims      = sort_dims
        self.allow_repeats  = allow_repeats
        self.no_overlaps    = no_overlaps
        self.region         = region
        self.dataset_index  = dataset_index
        self.key_label      = key_label
        self.is_required    = is_required

        # Store initialization parameter names for pickling. Note: attributes
        # added above this line will be included in the hash calculation that
        # determines cache naming (unless included in CACHE_KEY_EXCLUSIONS)
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
        """ Short-form representation """
        return f'Datafile[{self.label}]'


    def to_string(self, show_stats: bool = False) -> str:
        """Long-form representation.

        Parameters
        ----------
        stats : bool
            Whether or not to compute various statistics about features.

        Returns
        -------
        str
            String containing information about the underlying data.

        """
        s = [
            f'Datafile: {self.name} (Dataset[{self.dataset_index}])',
            f'Location: {self.location}',
            f'XArray: \n{self.data}'.replace('\n', '\n\t'),
        ]
        if show_stats:
            with pd.option_context('display.max_columns', None):
                stats = self.summary.to_dataset('features').to_pandas().T
                s += [f'Stats: \n{stats}'.replace('\n', '\n\t')]
        return '\n'.join(s)


    @cached_property
    def _raw_data(self):
        """ Only read the data once necessary """
        if isinstance(self.location, xr.Dataset):
            raw = self.location
        else:
            # If the given location isn't already an xr.Dataset, open it
            if (
                isinstance(self.location, FSMap)
                and not isinstance(self.location, S3Path)
            ):
                self.location = S3Path(self.location)
            raw = get_backend(self.location).open(**self._kwargs)

        # Keep the valid_mask only if it is still accurate for this Datafile
        if 'valid_mask' in raw and len(self.preprocessors) or len(self.extent):
            raw = raw.drop_vars(['valid_mask'], errors='ignore')

        # Handle the summary statistics
        if 'summary' in raw:

            # If no extents are changed / preprocessors applied, raw_data
            # statistics will be the same as data statistics
            if not (len(self.preprocessors) or len(self.extent)):
                self.__dict__['summary'] = raw['summary']
            raw = raw.drop_vars(['summary', 'features', 'statistics'],
                                errors='ignore')

            # Set datetime dtype back to np.datetime64 if it was converted
            if (
                hasattr(raw, 'datetime')
                and np.issubdtype(raw.datetime.dtype, np.float32)
            ):
                d64 = raw.datetime.astype('datetime64[m]')
                raw = raw.assign_coords(datetime=d64)

        # All coordinates must be in ascending order, with data chunked by dask
        return raw.sortby(list(raw.coords)).chunk({})


    @cached_property
    def region_mask(self):
        """ Use RegionalMaskGenerator to create mask for chosen region """
        # Import only when needed to avoid requiring gdal, rasterio, opencv
        from .RegionalMaskGenerator import RegionalMaskGenerator
        assert(self.region is not None), 'Must specify region for masking'
        return RegionalMaskGenerator(self.region).gen_mask(self._raw_data)


    @cached_property
    def data(self) -> xr.DataArray:
        """ Loaded xarray object """
        data = self._raw_data

        # Generate and apply mask based on region name
        if (self.region is not None):
            data = data.where(self.region_mask)

        # Apply any preprocessing functions
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            for func in self.preprocessors:
                data = func(self, data)

        # Select only requested features
        keys = sorted(self.features or data.keys())
        keys = keys + (['valid_mask'] if 'valid_mask' in data else [])
        data = data[keys]

        # Add virtual dimensions here, after dropping extra features (which may
        # have dropped some dimensions as a side-effect)
        missing = {k:v for k,v in self._virtual_dims.items() if k not in data}
        if missing:
            data = data.expand_dims(dim=missing)

        # Remove summary statistics
        data = data.drop_vars(['summary', 'features', 'statistics'],
                              errors='ignore')

        # Select only requested coordinates
        data = data.sel({k: slice(*ext) for k, ext in self.extent.items()
                        if k in data and k not in self._virtual_dims})

        # Store original data dtypes - all coords are converted to float in
        # Datafile._typed_data. In order to allow datetime64 to fit float32,
        # we convert datetime values to datetime64[m] (i.e. minute resolution)
        dtypes = {
            str(k): T if not np.issubdtype(T := data[k].dtype, np.datetime64)
            else 'datetime64[m]' for k in list(data)+list(data.coords)
        }

        # Create a mask for valid data elements, to pre-compute when caching
        if 'valid_mask' not in data:
            is_sparse = lambda f: hasattr(type(data[f].data._meta), 'todense')
            if not any(map(is_sparse, data)):
                valid_mask = ~data.to_array('features').isnull()

                # If at least some part of a window needs to be valid, all
                # features must be simultaneously valid
                if np.prod(list(self._valid_percent.values())) > 0:
                    data['valid_mask'] = valid_mask.all('features')

                # If empty windows are allowed, any features can be valid
                else:
                    data['valid_mask'] = valid_mask.any('features')

        if 'valid_mask' in self.features:
            self.features.remove('valid_mask')

        # Only rectilinear (separable-coordinate) grids are currently supported
        for name, coord in data.coords.items():
            n_dims = len(list(coord.dims))
            if n_dims > 1:
                err = f'{self}.{name} has {n_dims} dims ({coord.dims})'
                raise ValueError(f'Only rectilinear grids are supported: {err}')

        # Convert to a DataArray and ensure data is backed by dask
        data = data.to_array('features').chunk({})

        # Add Datafile configuration to the attributes
        # if 'Datafile.config' not in data.attrs:
        data.attrs.update({
            'Datafile.config'      : self.config,
            'Datafile.config_hash' : self.config_hash,
        })

        # Save the original coordinates/dtypes for later return values
        original_dims = [c for c in data.coords if c not in self._virtual_dims]
        self.original_dims = (
            sorted(original_dims) if self.sort_dims else original_dims,
            [f for f in data.features.to_numpy() if f not in ['valid_mask']],
            dtypes,
            [f for f in self.features if f not in ['valid_mask']],
        )

        # Sanity check - Sample.py assumes features are sorted
        assert(sorted(self.original_dims[1]) == list(self.original_dims[1])), \
            self.original_dims[1]

        # Transpose dimensions so they are in the correct order, and return
        order = sorted(set(data.coords) - {'features'})
        return data.transpose(*order, ...)


    @cached_property
    def summary(self) -> xr.DataArray:
        """ Return summary statistics for data """
        # Ensure cached summary data is used if available
        self._raw_data  # noqa: B018
        if 'summary' in self.__dict__:
            return self.__dict__['summary']
        data = self.data.drop_sel(features='valid_mask', errors='ignore')
        data = data.to_dataset('features')

        # Splitting large chunks seems to sometimes result in KeyError in dask
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):

            # Include in the summary stats any coordinates requested as features
            for coord in data.coords:
                if (coord in self.features) and (coord != 'datetime'):
                    coords = {c: data[c] for c in data.coords if c != coord}
                    data[f'{coord}_f'] = data[coord].astype('float32')
                    data[f'{coord}_f'] = data[[f'{coord}_f']] \
                        .expand_dims(**coords)[f'{coord}_f'] \
                        .transpose(*list(data.dims)) \
                        .chunk(self.data.data.chunksize[:-1])

            # Slightly different handling required for sparse data
            if hasattr(type(self.data.data._meta), 'todense'):
                def _calc(values):
                    d_append = dask.delayed(np.append)
                    d_series = dask.delayed(pd.Series)

                    # Explicitly apply functions per-block to avoid densifying
                    blocks = values.data.to_delayed().ravel()

                    # Ensure there's always at least one element to avoid error
                    delays = [d_append(b.data, [np.nan]) for b in blocks]
                    series = [d_series(b) for b in delays]

                    # Fake array sizes so dask doesn't complain
                    length = [1] * len(series)
                    series = df.from_delayed(series)
                    arrays = series.to_dask_array(lengths=length)
                    # Alternatively:
                    #  values.data.map_blocks(lambda b: b.data[:,None,None])

                    # Similar to dense summary, but use dask ops directly
                    extra = dd(int, {
                        'null' : lambda x: values.size - da.sum(da.isfinite(x)),
                    })
                    stats = ['mean', 'std', 'min', 'max'] + list(extra)
                    coord = xr.Variable('statistics', stats)

                    nan_f = lambda k: getattr(da, f'nan{k}', extra[k])
                    value = lambda k: nan_f(k)(arrays).astype('float32')
                    toset = lambda k: xr.DataArray(value(k))
                    array = xr.concat(map(toset, stats), coord)
                    stats = array.to_dataset('statistics')

                    q_to_name = lambda q: f'p{q}'.replace('p50', 'median')
                    quantiles = list(range(1, 100))
                    key_names = list(map(q_to_name, quantiles))
                    stats[key_names] = da.percentile(arrays, quantiles,
                                                     internal_method='tdigest')
                    return stats.to_array('statistics')
                stats = data.apply(_calc)

            # Dense summary computation relies mostly on xarray operations
            else:
                extra = dd(int, {
                    'null' : lambda: data.isnull().sum(dtype='int64'),
                })
                stats = ['mean', 'std', 'min', 'max'] + list(extra)
                coord = xr.Variable('statistics', stats)

                get_f = lambda k: getattr(data, k, extra[k])()
                value = lambda k: get_f(k).to_array('features')
                array = xr.concat(map(value, stats), coord)
                stats = array.to_dataset('statistics')

                # Compute percentiles as a group for efficiency
                q_to_name = lambda q: f'p{q}'.replace('p50', 'median')
                quantiles = list(range(1, 100))
                key_names = list(map(q_to_name, quantiles))
                stats[key_names] = xr.apply_ufunc(
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


    # def summary(self, compute: bool = True) -> xr.DataArray:
    #     """ Summary re-write which is simpler, but uses more memory / time
    #          in large-data scenarios """
    #     # Ensure cached summary data is used if available
    #     self._raw_data
    #     if 'summary' in self.__dict__:
    #         return self.__dict__['summary']

    #     data = self.data.to_dataset('features')
    #     desc = f'{self} stats'
    #     summ = []
    #     for k in tqdm(self.features, desc=desc, disable=not compute):
    #         if k == 'datetime': continue

    #         darray = da.asarray(data[k].data.ravel())
    #         finite = da.isfinite(darray)
    #         values = darray[finite].compute_chunk_sizes()
    #         pctile = da.percentile(values, [25,50,75],
    #                                internal_method='tdigest')

    #         stat = [('null', darray.size - da.sum(finite))]
    #         stat+= [(s, getattr(da,f'nan{s}')(values))
    #                 for s in ['mean','std','min','max']]
    #         stat+= [(s, pctile[j]) for j, s in enumerate(
    #             ['p25', 'median', 'p75']
    #         )]

    #         # Compute features independently for efficiency
    #         keys, vals = map(list, zip(*stat))
    #         vals = list(da.compute(*vals)) if compute else da.stack(vals)
    #         summ.append((k, xr.DataArray(vals, {'statistics':keys})))

    #     self.__dict__['summary'] =xr.Dataset(dict(summ)).to_array('features')
    #     return self.__dict__['summary']


    @property
    def _typed_data(self) -> xr.DataArray:
        """ Data/coords converted to float types """
        data = self.data.drop_sel(features='valid_mask', errors='ignore')

        # Cast int/uint/etc. to float in order to allow NaNs
        # for key in data.features:
        if (
            np.issubdtype(data.dtype, np.number)
            and not np.issubdtype(data.dtype, np.floating)
        ):
            data = data.astype(np.promote_types(data.dtype, np.float16))

        # Coordinates must have uniform type, so we cast to float32
        # Note: datetime64 are converted to datetime64[m] (minute resolution)
        for key in data.coords:
            if key != 'features':
                f_64 = lambda d: d.astype('datetime64[ns]').astype('float64')

                if np.issubdtype(data[key].dtype, np.datetime64):
                    data = data.assign_coords({key: f_64(data[key]) / 6e10})
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
        return da.stack(c_grid, axis=-1)  # .rechunk({-1:-1})


    @property
    def name(self) -> str:
        """ Return a name for this Datafile using the location if possible """
        if self.key_label:
            path = Path(self.key_label)

        # Any locations that are Path-like
        elif isinstance(self.location, FSMap):
            path = Path(self.location.root)
        elif isinstance(self.location, str):
            path = Path(self.location)
        elif isinstance(self.location, (Path, S3Path)):
            path = self.location

        # Any other type (like directly passed xarray objects)
        else:
            return f'{type(self.location).__name__}_{self.config_hash[:4]}'

        # Name is just the stem of the path, plus the parent if cached
        name = path.stem
        if len(name) == len(self.config_hash):
            name = f'{path.parent.stem}_{name[:4]}'
        return name


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
            try:
                code = hashlib.sha256(f.__code__.co_code).hexdigest()
            except Exception as e:
                code = e
            return f'{name}: {code}'

        config = dict(self.__getstate__())
        config['preprocessors'] = list(map(f_repr, config['preprocessors']))
        return {k: str(v) for k,v in config.items()}


    @property
    def config_hash(self) -> str:
        """ Hash of the config dictionary that is used as a condensed label """
        config = str(dissoc(self.config, *self.CACHE_KEY_EXCLUSIONS))
        return hashlib.sha256(config.encode('utf-8')).hexdigest()


    @property
    def data_features(self) -> list[str]:
        """ Feature names contained in the xarray.Dataset object """
        return sorted(set(self.data.features.values) - {'valid_mask'})


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
        return {}


    @property
    def virtual(self) -> np.ndarray:
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


    def sparsity(self, compute=False):# -> da.Array | float:
        """ Create dask object that computes percentage of invalid data """
        null = self.summary.sel(statistics='null')
        data = self.data
        if 'valid_mask' in data.features:
            data = data.drop_sel(features='valid_mask')
        if 'valid_mask' in null.features:
            null = null.drop_sel(features='valid_mask')
        sparsity = null.data.sum() / data.size
        return sparsity.compute() if compute else sparsity


    def is_sparse(self, compute=True, threshold=0.99):
        """ Datafile is deemed sparse if sparsity > 99% """
        return hasattr(type(self.data.data._meta), 'todense')
        # return self.sparsity(compute) > threshold


    @property
    def valid_percent(self) -> dict[str, Number]:
        """ {Dimension key : valid percent} with default value for missing.
            Keys are also expanded into tuples, and missing dims are added. """
        default = Datafile.DEFAULT_VALID_PERCENT
        expand  = lambda k: tuple(np.atleast_1d([k]).flatten())
        mapping = {expand(dim): v for dim, v in self._valid_percent.items()}
        defined = np.hstack(mapping | {'':''})
        mapping|= {(dim,): default for dim in self.dims if dim not in defined}
        return dd(lambda: default, mapping)


    @property
    def window_depth(self) -> dict[str, np.ndarray]:
        """ {Dimension key : window depth} with default value for missing.
            Values are also expanded into (left side, right side) formats. """
        def handle_None(dim, size):
            """ None given for a dim indicates using the full dimension """
            if size is None:
                # Even-sized dims use window=(half, half-1) to handle center
                size = self.data[dim].size
                left = size // 2
                return left, (size - left) - 1
            return size

        flatten = lambda v: np.array([v]).flatten().astype(int)
        expand  = lambda v: np.array([flatten(v)[0], flatten(v)[-1]])
        windows = {d: handle_None(d,v) for d,v in self._window_depth.items()}
        mapping = valmap(expand, windows)
        return dd(lambda: expand(Datafile.DEFAULT_WINDOW_SIZE), mapping)


    @property
    def window_total(self) -> dict[str, Int]:
        """ Window total size per dimension """
        return {dim: 1+self.window_depth[dim].sum() for dim in self.dims}


    @property
    def invalid_value(self) -> list[Number]:
        """ Invalid values, including any defined in the defaults """
        user_defined = np.atleast_1d(self._invalid_value).tolist()
        return user_defined + Datafile.DEFAULT_INVALID_VALUES


    @property
    def numblocks_dict(self) -> dict[str, Int]:
        """ Dict of {dim: numblock} for all dimensions """
        return dict(zip(self.dims, self.numblocks[:-1], strict=True))


    @property
    def block_minim(self) -> dict[str, Int]:
        """ Minimum number of coordinates allowed in a block, per dimension """
        # If this Datafile isn't required, its blocks are allowed to be empty
        # Otherwise, blocks need at least enough elements to fill the window
        return valmap(lambda w: w*self.is_required, self.window_total)


    @cached_property
    def is_uniform(self) -> bool:
        """ Flag indicating whether coordinates are uniform grids """
        self.resolution  # noqa: B018  # Sets the is_uniform flag
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

        if self.is_sparse() and not self.match_radius:
            raise ValueError(f'{self} Must set match_radius for sparse dataset')

        # Adjust the resolution vector based on the requested match radius
        for dim, radius in self.match_radius.items():
            assert(dim in self.dims), f'Unknown dimension "{dim}"'
            index = self.dims.index(dim)

            # Handle radius specified as a percentage
            # The default radius used by the neighbor matching is 50%, so we
            # also need to scale the new radius by that amount for it to align
            if isinstance(radius, str):
                assert(radius[-1] == '%'), \
                    f'Unknown radius specification "{radius}"'
                vectors[index] *= (float(radius[:-1])/100.) / 0.5

            # Handle radius specified as a raw value (datetime is minutes)
            else: vectors[index][:] = radius * 2

        # Stack the left/right resolution for each vector
        setattr(self, 'is_uniform', all(map(uniform, vectors)))
        if not self.is_uniform:
            stack_lr = lambda i, vector: np.stack([
                np.r_[vector[:1], vector],
                np.r_[vector, vector[-1:]]
            ], axis=-1) if not self.virtual[i] else np.array([[np.nan, np.nan]])
            return list(map(stack_lr, *zip(*enumerate(vectors))))
        return [vec.min() for vec in vectors]


    @cached_property
    def max_resolution(self) -> list:
        """ Return the maximum resolution for each coordinate.
        If we have uniform grids, this will just be the resolution
        itself; otherwise it will be the maximum along each vector """
        # return [np.atleast_1d(r).max() for r in self.resolution]

        # Sparse Datafiles should be excluded from max_resolution calculations
        if self.is_sparse(): return [0] * len(self.dims)

        # We don't want to apply any manual match_radius however, so we redo
        # the resolution calculations without applying match_radius
        get_vec = lambda d: np.diff(d) if len(d) > 1 else np.zeros(1)
        vectors = [get_vec(self._typed_data[dim]).round(5) for dim in self.dims]
        return [np.atleast_1d(r).max() for r in vectors]


    def ensure_dims(self, dims: set[str]) -> None:
        """ Ensure any missing dimensions are added as virtual dimensions """
        missing = dims - set(self.dims)

        if missing:
            # Add the new dimension(s) to the raw data and valid_percents
            missing_to_nan = lambda m: dict(zip(m, [[np.nan]]*len(m)))
            missing_in_raw = missing_to_nan(missing - set(self._raw_data.dims))
            missing_in_dat = missing_to_nan(missing)

            # raw data can have dimensions dropped when features are selected,
            # whereas the full self._raw_data object still contains those dims
            if missing_in_raw:
                self._raw_data = self._raw_data.expand_dims(dim=missing_in_raw)

            # Add the new dimension(s) to virtual_dims to track numblocks
            self._virtual_dims.update(missing_in_dat)
            self.dims = sorted(dims)

            # Force a cache refresh for these values
            for key in ['data', 'valid_percent']:
                self.__dict__.pop(key, None)


    @property
    def max_valid_blocks(self) -> list[Number]:
        """ Return the maximum number of blocks along each dimension.

        The max blocks along each dimension which would create a valid
        configuration depends on the requested window depths and the
        valid percent (the percent which must be valid for a window
        sample to be considered valid).

        In general, the minimum chunk size allowed corresponds to the
        requested window depth on each side. For example, window depths
        of (10, 5) means a data chunk configuration of (10, 1, 5) would
        be technically be valid - and therefore data with 16 elements
        could maximally be split into 3 blocks. Dask requires chunks to
        be the same size (except the last), however, and so the maximum
        configuration is two blocks with chunksizes (10, 6).

        This scenario also assumes a valid percent of 1 (100% of elements
        must be valid for a window to be valid). If we take into account
        different valid percents, say, 0.5 (50%), the minimum total chunk
        size (left + center + right) is then halved.

        In summary we have two primary constraints:
            - chunks to the left and right of every chunk must be at least
              as large as the requested left/right window depth
            - total chunk size must be at least as large as the total window
              size, multiplied by the valid percent

        Returns
        -------
        list[Number]
            List of numbers indicating the maximum number of blocks
            that can be created along each dimension for the current
            Datafile objects contained in this Dataset.

        """
        blocks = []
        for i, dim in enumerate(self.dims):

            # Virtual dimensions can have any number of blocks
            if self.virtual[i]:
                blocks.append(np.inf)
                continue

            # Each dimension must be at least as large as the minimum
            # number of elements required for a valid window
            size  = self.shape[i]
            total = self.window_total[dim]
            v_pct = self.valid_percent[dim]
            # depth = self.window_depth[dim]

            if size < (total * v_pct):
                raise ValueError(f'{self} (shaped {self.shape}) has {size} ' +
                    f'elements along dimension "{dim}", which is less than ' +
                    f'the required size of {total*v_pct} for a window size of'+
                    f' {total} ({self.window_depth[dim]}) and a valid percent'+
                    f' of {v_pct} ({int(v_pct*100)}%)')

            min_chunksize = int(max(1, max(self.window_depth[dim])))
            max_numblocks = int(size // min_chunksize)
            # last_chunksize = int(size % min_chunksize)

            # # Dask allows the last chunk to be smaller, and so technically
            # # we could have one additional block if we allow the remainder
            # # to be smaller than the minimum chunk size (so long as it's
            # # greater than the right window depth). However, this requires
            # # careful handling of the chunking process, and is more likely
            # # to cause bugs than to provide any significant benefit
            # if (last_chunksize > 0) and (last_chunksize >= depth[1]):
            #     max_numblocks += 1
            blocks.append(max_numblocks)
        return blocks


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
        def _calculate(dim, res, max_res, skip, blocks, max_chunksize):
            if not self.is_uniform:
                # Using mean/median here would be better for performance,
                # but could miss some matches due to too little overlap
                res = res.min()

            if blocks > 1:
                # total = self.window_depth[dim].sum()
                size  = self.window_depth[dim].max()
                size += bool(res) if skip or (res==0) else ((max_res/2) / res)
            else: size = 0

            # Clip overlap to max chunksize, since it can only span one block
            overlap = min(max_chunksize, math.ceil(np.nan_to_num(size)))
            return (self.dims.index(dim), overlap)

        ndim = len(self.dims)
        res = np.zeros((ndim, 1)) if self.is_sparse() else self.resolution
        blks = dimension_blks if dimension_blks is not None else [2] * ndim
        args = [self.dims, res, max_resolution, skip_dimension, blks]
        args+= [list(map(max, self.dask.chunks[:-1]))]

        if len(set(map(len, args))) > 1:
            raise ValueError(f'Args not all the same length: {args}')
        return dict(map(_calculate, *args))


    def update_blocks(self, numblocks: Collection, verify:bool = True) -> list:
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
        verify : bool
            Whether to require the new block structure is exactly equal to the
            requested number of blocks. This isn't always the case, as zarr
            requires all chunks (besides the last) to have the same size - and
            so e.g. a dimension with length 7 cannot be split into 5 blocks as
            it would result in the chunks [2,2,1,1,1].

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
        extra_dim = len(self.dask.numblocks) - (len(numblocks)+1)
        numblocks = list(numblocks) + [1] * extra_dim

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

        newchunks = [math.ceil(shape / block) for block, shape in block_shape]
        if self.numblocks[:-1] != tuple(numblocks):
            self.data = self.chunk(dict(zip(self.dims, newchunks)))
        else: newchunks = []

        # Ensure the new block numbers are equal to what was requested
        if any(block != db for block, db in zip(numblocks, self.numblocks)):
            newchunks = [shape // block for block, shape in block_shape]
            self.data = self.chunk(dict(zip(self.dims, newchunks)))

            if any(block != db for block,db in zip(numblocks, self.numblocks)):
                m = f'{self}: request={numblocks} result={self.numblocks[:-1]}'
                if verify:
                    raise ValueError(m)
                print(f'WARNING - {m}')
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
        extra_dim = len(self.dask.chunks) - (len(chunksize)+1)
        chunksize = list(chunksize) + [-1] * extra_dim

        # Update virtual_dims to track the requested number of blocks
        is_virtual = lambda dim: dim[0] in self._virtual_dims
        calc_block = lambda s,c: getattr(c, '__len__', lambda: max(s//c, 1))()
        dim_chunks = zip(self.dims, chunksize)
        dim_block  = zip(self.dims, map(len, map(np.atleast_1d, chunksize)))
        self._virtual_dims.update(dict(filter(is_virtual, dim_block)))
        self._target_blocks = np.array(list(
            map(calc_block, self.shape, chunksize))
        )

        chunksize   = [1 if is_virtual([dim]) else n for dim, n in dim_chunks]
        block_shape = list(zip(chunksize, self.shape))

        # Check that all dimensions are at least as large as the requested size
        # if any(block > shape for block, shape in block_shape):
        for block, shape in block_shape:
            message = f'Blocks={chunksize} != data.shape={self.shape}'
            if hasattr(block, '__len__'):
                if sum(block) != shape:
                    raise ValueError(message)
            else:
                if block > shape:
                    raise ValueError(message)

        self.data = self.chunk(dict(zip(self.dims, chunksize)))
        return chunksize


    def apply_overlap(self,
        overlaps    : dict[Int, Int] | Int,
        boundary    : dict[Int, Number] | Number | str = np.nan,
        optimize    : bool = True,
        cache_path  : Path | S3Path | None = None,
        overwrite   : bool = False,
        task_blocks : int = 256,
        trim_every  : int = 8,
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

        # def dask_overlap(x, depth, boundary, *, allow_rechunk=False):
        #     """ Rewrite dask.overlap.overlap to remove forced rechunking """
        #     if all(v == 0 for v in depth.values()): return x
        #     from dask.array.overlap import coerce_depth, coerce_boundary
        #     from dask.array.overlap import boundaries, overlap_internal
        #     depth = coerce_depth(x.ndim, depth)
        #     bound = coerce_boundary(x.ndim, boundary)
        #     x2 = boundaries(x, depth, bound)
        #     x3 = overlap_internal(x2, depth)
        #     over = lambda k: int(bound.get(k, "none") != "none")
        #     trim = {k: v*2*over(k) for k,v in depth.items()}
        #     return da.chunk.trim(x3, trim)

        def get_id(index):
            """ Get the block id from the multi-index representation """
            return np.ravel_multi_index(index, tuple(blocks) + (1,))

        def set_id(array):
            """ Set all values in a block equal to its own block id """
            f = lambda x, block_id: np.full_like(x, get_id(block_id),
                                                 dtype='int32')
            return array.map_blocks(f, dtype='int32')

        def mask_id(array):
            """ Boolean mask indicating if a value originated in this block """
            f = lambda x, block_id: x != get_id(block_id)
            return array.map_blocks(f, dtype=bool)

        # Ensure name=False to avoid hashing all values in the array
        masks_kwargs = {'dtype': bool, 'name': False, 'chunks': chunks+((1,),)}

        # Create mask indicating out-of-bound elements in each block
        inbound_mask = da.ones(shapes + (1,), **masks_kwargs)

        # Create mask indicating overlapped elements in each block
        overlap_mask = set_id( da.zeros(shapes + (1,), **masks_kwargs) )

        if self.no_overlaps or (optimize and self.is_sparse()):
            overlaps = 0

        # Clip block extents to avoid duplication
        # extents = [max(c)+o*2 for c,o in zip(self.chunks, overlaps.values())]

        # Convert overlaps to a dictionary for each dimension if only int given
        if isinstance(overlaps, int):
            overlaps: dict = {d: overlaps for d in range(len(shapes))}

        # Clip overlaps to max chunk size, since they can only span one block
        overlaps = {d: min(max(chunks[d]), o) for d, o in overlaps.items()}

        # Create overlapping blocks, tiling virtual dimensions where necessary
        virtual = [self._virtual_dims.get(d, 1) for d in self.dims] + [1]
        repeat  = partial(da.tile, reps=virtual)
        daskopt = lambda x: x  # dask.optimize(x)[0] if optimize else x
        blocker = lambda x: daskopt(repeat(x)).blocks
        overlap = partial(dask_overlap, **{
            'depth'            : overlaps,
            'boundary'         : boundary,
            'allow_rechunk'    : False,
            'blocks_per_batch' : task_blocks,
            'trim_every'       : trim_every,
        })

        def _rechunk(v):
            if v.chunks != chunks:
                v = v.rechunk(chunks)
            return v[..., None]

        # valid_mask = da.isfinite(self.dask).all(axis=-1, keepdims=True)
        # valid_mask = self.data.sel(features=['valid_mask']).data.astype(bool)
        raw = self._raw_data.transpose(*sorted(self._raw_data.coords))
        raw = raw.sel({k: slice(*ext) for k, ext in self.extent.items()
                        if k in raw and k not in self._virtual_dims})

        if 'valid_mask' in raw:
            valid_mask = _rechunk(raw['valid_mask'].data)
        else: valid_mask = da.isfinite(self.dask).all(axis=-1, keepdims=True)

        # TODO: Gracefully switch to fallback method if this is not cached
        assert(len(self.preprocessors) == 0), 'Need to cache dataset first'

        # Use data from original database to allow direct reading from files
        # (xarray combines features in self.data so that they are inseparable)
        # Note that this ONLY works for cached data, where preprocessors have
        #  already been applied
        feat_names = list(self._typed_data.features.values)
        feat_array = [_rechunk(raw[k].data) for k in feat_names]

        # Separate data/coords/masks into independent blocks
        block_grids = list(map(blocker, list(map(overlap, feat_array)) + [
            # overlap(self.dask),
            overlap(self.dask_coords),
            overlap(inbound_mask, boundary=0),
            mask_id(overlap(overlap_mask, boundary=-1)),
            overlap(valid_mask, boundary=0),
        ]))

        # No overlaps means the overlap mask is just zero
        if all(v == 0 for v in overlaps.values()):
            block_grids[-2] = blocker(da.zeros(shapes+(1,), **masks_kwargs))

        # Define args/kwargs for the Block objects that will be created
        block_kwarg = {
            'dims'          : self.dims,
            'original_dims' : self.original_dims,
            'window_depth'  : self.window_depth,
            'valid_percent' : self.valid_percent,
            'invalid_value' : self.invalid_value,
            'allow_repeats' : self.allow_repeats,
            'label'         : self.label,
            'sparsity'      : self.sparsity(optimize),
            'key_label'     : self.key_label,
        }

        # Ensure we're generating the same number of blocks for all grids
        is_equal = lambda a: np.prod(a) == np.prod(blocks)
        b_shapes = [block.shape[:-1] for block in block_grids]
        if not all(map(is_equal, b_shapes)):
            if any(b < 1 for b in blocks):
                raise ValueError(f'Negative number of target {blocks=}')
            raise ValueError(f'Not all grids have {blocks=}: {b_shapes}')

        # Cache final data blocks to zarr
        if cache_path is None:
            cache_path = getattr(self, '_cache_path', None)

        # Sparse (tiledb) datafiles are not currently able to be cached
        if cache_path is not None and 'tiledb' not in self.cache_name:
            block_grids = self._cache_blocks(
                cache_path, block_grids, feat_names, overwrite,
            )

        get_value = lambda i, d: d[i] if isinstance(d, dict) else d
        to_blocks = lambda i, a: da.tile( dask_overlap(a, **{
            'depth'            : {0: get_value(i, overlaps)},
            'boundary'         : {0: get_value(i, boundary)},
            'allow_rechunk'    : False,
            'blocks_per_batch' : task_blocks,
            'trim_every'       : trim_every,
        }), reps=[virtual[i]] + ([1]*(a.ndim-1)) ).blocks

        def vectors_to_blocks(vectors, chunks):
            """ From the given vectors create blocks that match data blocks """
            # Create a BlockView object for each vector
            to_arr = partial(da.from_array, name=False)
            arrays = map(to_arr, vectors, chunks)
            blocks = list(starmap(to_blocks, enumerate(arrays)))
            counts = [b.size for b in blocks]
            assert(is_equal(counts)), [counts, blocks]

            # Wrap the vectors to be indexable as a cartesian product array
            fetch = lambda _, idx: tuple(b[i] for b,i in zip(blocks, idx))
            return type('product_array', (object,), {'__getitem__': fetch})()

        # Non-uniform coordinate grids use different resolutions in each block
        if not self.is_uniform:
            r_chunks = [[chunk, (2,)] for chunk in chunks] # 2 = left/right
            r_vector = self.resolution
            r_blocks = vectors_to_blocks(r_vector, r_chunks)
            block_grids.append(r_blocks)

        # Uniform resolution coordinates use the same resolution across blocks
        else: block_kwarg['resolution'] = self.resolution

        # Create blocks for coordinate vectors
        c_chunks = [[chunk] for chunk in chunks]
        c_vector = [self._typed_data[d].values for d in self.dims]
        c_blocks = vectors_to_blocks(c_vector, c_chunks)

        # Separate features into their own list
        num_feats = len(feat_array)
        sep_feats = lambda i: (
            [[g[i] for g in block_grids[:num_feats]]] +
             [g[i] for g in block_grids[num_feats:]]
        )

        # Returns list of functions to allow lazy creation of the Block objects
        gen_block = lambda i: (lambda: Block(*sep_feats(i), **block_kwarg | {
            'coord_vecs' : c_blocks[i],
        }))
        lazy_func = gen_block  # lambda i: cache(lambda: gen_block(i))
        return map(lazy_func, product(*map(range, blocks)))


    def reset(self, **kwargs) -> None:
        """ Reset Datafile state to remove cached properties.

        Parameters
        ----------
        **kwargs
            Any __init__ parameters that should be updated in the config state.

        """
        config = self.__getstate__() | kwargs
        self.__dict__.clear()
        self.__dict__.update(config)


    def copy(self, **kwargs) -> Datafile:
        """ Return a deepcopy of the current Datafile.

        Parameters
        ----------
        **kwargs
            Any __init__ parameters that should be updated in the config state.

        Returns
        -------
        Datafile
            A deepcopy of the current Datafile, updated with any given kwargs.

        """
        config = copy.deepcopy(self.__getstate__())
        config.pop('_init_keys', None)

        # Fix keys that have their init value made private
        config = {k.removeprefix('_'): v for k,v in config.items()}
        config|= config.pop('kwargs')
        return self.__class__(**(config | kwargs))


    @property
    def cache_name(self) -> str:
        """ Name of the folder this Datafile would be cached in """
        ext = '.tiledb' if '.tiledb' in str(self.location) else '.zarr'
        return Path(self.name, f'{self.config_hash}{ext}').as_posix()


    def _cache(self,
        overwrite : bool = False,
        verbose   : bool = False,
        cache_dir : Path | str | FSMap | S3Path = 'Cache',
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
        verbose : bool
            Whether to print more detailed info when caching this datafile.
        cache_dir : Path | str
            Location for the cached data to be stored. By default, data is
            cached in `./Cache`.

        """
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        elif isinstance(cache_dir, FSMap):
            cache_dir = S3Path(cache_dir)

        message = f'Cache exists for {self.name} {self.config_hash[:6]}..'
        timer = Stopwatch(message, silent=True)
        timer.__enter__()

        data = self.data.to_dataset('features')
        dest = cache_dir.joinpath(self.cache_name)

        if isinstance(dest, Path):
            dest.parent.mkdir(exist_ok=True, parents=True)

        # Erase virtual dimensions
        for dim in data.coords:
            if dim in [d for d,v in zip(self.dims, self.virtual) if v]:
                data = data.isel({dim: 0}, drop=True)

        # Include summary statistics
        data['summary'] = self.summary
        if 'valid_mask' in data:
            data['valid_mask'] = data['valid_mask'].astype(bool)

        # Include resolution features if available
        if 'resolution_lat_max' in self._raw_data:
            slices = lambda d: slice(d.min().item(), d.max().item())
            extent = {k: slices(data[k]) for k in ['latitude', 'longitude']}
            chunks = {k: data.chunks[k]  for k in ['latitude', 'longitude']}

            for k in ['lat', 'lon']:
                for m in ['min', 'max']:
                    key = f'resolution_{k}_{m}'
                    data[key] = self._raw_data[key].sel(extent).chunk(chunks)

        # Allow tracking the reason a datafile is re-cached
        reason = 'user passed overwrite=True to _cache'
        backend = get_backend(dest)

        # Verify cached data is equivalent
        if (not overwrite) and dest.exists():
            try:
                cache = backend.open()
                attrs = ['dims', 'chunksizes']

                # Temporary check for updated attrs format
                if 'cache_uid' not in cache.attrs:
                    reason = 'Update cache attrs'
                    overwrite = True

                for a in attrs:
                    curr = getattr(data,  a, None)
                    prev = getattr(cache, a, None)

                    # Ignore xarray dims future warning
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=FutureWarning)
                        if curr != prev:
                            # curr = reprlib.repr(curr) # f'{curr=}'[:80]
                            # prev = reprlib.repr(prev) # f'{prev=}'[:80]
                            reason = f'differing {a}\n\t{curr=}\n!=\n\t{prev=}'
                            overwrite = True
                            break
                else:
                    if not bool(xr.align(data, cache, **{
                        'join'    : 'exact',
                        'exclude' : 'statistics',
                    })):
                        reason = 'misaligned cache'
                        overwrite = True

            except KeyboardInterrupt:
                raise

            except Exception as e:
                reason = f'exception {e}'
                overwrite = True

        if overwrite or (not dest.exists()):
            logger = get_logger(__name__)
            logger.info(f'Caching {self.name} to {dest}...')

            if dest.exists():
                logger.info(f'Re-caching reason: {reason}')
            if verbose:
                logger.info(self.to_string())

        # Reinitialize this Datafile with the new cache
        # Anything handled by the cache (e.g. extent) can be dropped
        self.reset(**{
            'location'      : dest,
            'extent'        : {},
            'preprocessors' : [],
            'region'        : None,
        })

        # Write the data to the destination
        if overwrite or (not dest.exists()):
            if dest.exists():
                if isinstance(dest, Path):
                    shutil.rmtree(dest)
                elif isinstance(dest, S3Path):
                    dest.delete()

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                # Add a unique random identifier for this specific cache
                rng = np.random.default_rng()
                val = str(rng.integers(1e8)).encode('utf-8')
                data.attrs['cache_uid'] = hashlib.sha256(val).hexdigest()

                try:
                    # Put coordinates into single chunk
                    backend.cache(dest, data, **{
                        'stream'   : sys.stderr,
                        'mode'     : 'w',
                        'encoding' : {
                            c: {'chunks': (-1,)} for c in data.coords
                        },
                    })
                except:
                    if dest.exists():
                        if isinstance(dest, Path):
                            shutil.rmtree(dest)
                        elif isinstance(dest, S3Path):
                            dest.delete()
                    raise

                # Ensure heap memory is released
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except FileNotFoundError: pass
        else:
            timer.silent = False
            timer.__exit__()


    def _cache_blocks(self,
        cache_path,
        block_grids,
        features,
        overwrite   : bool = False,
        verbose     : bool = False,
    ):
        """ Cache overlapped Block data in zarr format for faster batching """

        self._cache_path = cache_path
        message = f'Cache exists for {self.name} {self.config_hash[:6]}..'
        timer = Stopwatch(message, silent=True)
        timer.__enter__()

        if '_cache_path' not in self._init_keys:
            self._init_keys += ['_cache_path']

        dest = cache_path.joinpath(self.cache_name)
        if isinstance(dest, Path):
            dest.parent.mkdir(exist_ok=True, parents=True)

        # Get chunk sizes for all block grids (discarding features dim at -1)
        arrays = [b._array for b in block_grids]
        chunks = [a.chunks[:-1] for a in arrays]

        # Get the largest chunk size along each dimension for each block
        largest = [map(max, c) for c in chunks]

        # Zip respective dims to get sets of unique sizes per dimension
        uniques = list(map(set, zip(*largest)))

        # All block grids must have the same chunks: one unique size per dim
        assert(all(len(dim_sizes) == 1 for dim_sizes in uniques)), uniques

        # Extract the sole unique value per dim to get the final padding sizes
        padding = tuple([dim_sizes[0] for dim_sizes in map(list, uniques)])

        def pad(block, padding=padding):
            """ Add padding to chunks to ensure all are the same size """
            pads = [(0, need-size) for size, need in zip(block.shape, padding)]
            if block.size:
                fill = getattr(block, 'fill_value', np.nan) # Handle sparse.COO
                return np.pad(block, tuple(pads+[(0,0)]), constant_values=fill)
            return np.zeros(padding, dtype=block.dtype) * np.nan

        def unpad(block, size):
            """ Remove the previously added padding from the chunks """
            size = np.array(size).flatten()
            return block[tuple([slice(None, int(s)) for s in size])]

        # Feature info: number of features, and feature chunk size
        f_info = [(str(a.shape[-1]), a.chunksize[-1:]) for a in arrays]

        # Pad the chunks to ensure they are all the same exact size
        padded = [(self.dims+[n], a.map_blocks(pad, chunks=padding+size))
                    for (n, size), a in zip(f_info, arrays)]
        # padded = [dask.optimize(p)[0] for p in padded]

        # Include the initial chunk sizes as well, for later restoration
        initial = da.stack(np.meshgrid(*chunks[0], indexing='ij'), axis=-1)
        padded += [([f'_s_{i}' for i in range(initial.ndim)], initial)]

        # Dataset containing all grids, with chunks padded to be the same size
        dims = features + ['coords', 'inbound', 'overlap', 'valid']
        data = xr.Dataset(dict(zip(dims+['sizes'], padded)))
        data.attrs['cache_uid'] = self._raw_data.attrs['cache_uid']

        # Verify cached data is equivalent
        reason = 'user passed overwrite=True to _cache_blocks'
        if (not overwrite) and dest.exists():
            try:
                cache = xr.open_zarr(dest)
                attrs = ['sizes', 'dims', 'chunksizes', 'attrs']
                for a in attrs:
                    curr = getattr(data, a, None)
                    prev = getattr(cache, a, None)

                    # Ignore xarray dims future warning
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=FutureWarning)
                        if curr != prev:
                            # name = lambda n:n[:80]+('..' if len(n)>80 else '')
                            # curr = reprlib.repr(curr) # name(repr(curr))
                            # prev = reprlib.repr(prev) # name(repr(prev))
                            reason = f'differing {a}\n\t{curr=}\n!=\n\t{prev=}'
                            overwrite = True
                            break
                else:
                    if not bool(xr.align(data, cache, **{
                        'join'    : 'exact',
                        'exclude' : 'statistics',
                    })):
                        reason = 'misaligned cache'
                        overwrite = True
            except KeyboardInterrupt:
                raise
            except Exception as e:
                reason = f'exception {e}'
                overwrite = True

        # Write the data to the destination
        if overwrite or (not dest.exists()):
            logger = get_logger(__name__)
            logger.info(f'\nCaching {self.name} blocks to {dest}')
            if dest.exists():
                logger.info(f'Re-caching reason: {reason}')
            if verbose:
                logger.info(self.to_string())

            if dest.exists():
                if isinstance(dest, Path):
                    shutil.rmtree(dest)
                elif isinstance(dest, S3Path):
                    dest.delete()

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    # Put coordinates into single chunk
                    get_backend(dest).cache(dest, data, **{
                        'stream'   : sys.stderr,
                        'mode'     : 'w',
                        'encoding' : {
                            c: {'chunks': (-1,)} for c in data.coords
                        },
                    })

                # If writing fails for any reason, delete the partial cache
                except:  # noqa: E722
                    if dest.exists():
                        if isinstance(dest, Path):
                            shutil.rmtree(dest)
                        elif isinstance(dest, S3Path):
                            dest.delete()
                    raise

                # Ensure heap memory is released
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except FileNotFoundError: pass
        else:
            timer.silent = False
            timer.__exit__()

        # Load the zarr cache and remove the padding elements
        data = xr.open_zarr(dest)
        sizes = data['sizes'].values
        shape = (1,) * (len(sizes.shape)-1) + (sizes.shape[-1],)
        block_sizes = da.array(sizes).rechunk(shape)

        new_chunks = []
        for i in range(len(sizes.shape) - 1):
            slices = [0] * len(sizes.shape)
            slices[i] = slice(None)
            slices[-1] = i
            new_chunks.append(tuple(sizes[tuple(slices)]))

        get_chunks = lambda dim: tuple(new_chunks) + data[dim].chunks[-1:]
        return [da.map_blocks(unpad, data[d].data, block_sizes,
                  chunks=get_chunks(d)).blocks for d in dims]


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
        if (
            not isinstance(self.location, (xr.Dataset, S3Path, FSMap))
            and not Path(self.location).exists()
        ):
            raise FileNotFoundError(f'File not found: {self.location}')

        for key, extent in self.extent.items():
            if len(extent) != 2:
                message = f'{key} extent must be [start, end]: {extent}'
                raise ValueError(message)

        for key, window in self._window_depth.items():
            if (not isinstance(window, (Int, type(None)))) and (len(window)>2):
                message = f'{key} window must be (before, after): {window}'
                raise ValueError(message)

        for key, percent in self._valid_percent.items():
            if not (0 <= percent <= 1):
                message = f'{key} percent must be in range [0, 1]: {percent}'
                raise ValueError(message)
