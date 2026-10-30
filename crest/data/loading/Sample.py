from __future__ import annotations

from collections.abc import Collection, Iterator

import xarray as xr
import numpy as np
import warnings


# Inheriting from BaseAbstract nearly triples the time for loading batches
# e.g. ~190 Batches/sec -> ~70 Batches/sec
class Sample:
    """Class which holds all Datafile windows for a single location.

    Notes
    -----
    When a valid window is found across all Datafiles, a Sample is
    created which holds the raw data necessary to construct an xr.Dataset
    object for each Datafile.

    These xr.Dataset objects contain the window which was defined
    for each Datafile, respectively, when the search was performed
    (e.g. 3 x 3 x 2 window [latitude x longitude x time]).

    Each xr.Dataset object contains all variables from the respective
    Datafile object it represents, as well as the coordinates from
    which those variables were retrieved. The xarray representation is
    constructed lazily, only when a Datafile window is directly accessed.

    Parameters
    ----------
    data : Collection[dict]
        The collection of data windows, with exactly one per Datafile.
        The dictionary elements should each contain all information
        necessary to initialize the respective xarray.DataArray object
        (i.e. data, coords, and dims, with one dim named 'features').
    dtype : object
        Numpy dtype used to represent this Sample in object arrays.

    """
    def __init__(self, data: Collection[dict], dtype=object):
        self.key_label = []
        self.container = []
        self.dtype = dtype
        self.cache = {}

        # Separate Sample metadata from the raw xarray constructor arguments
        for i,item in enumerate(map(dict, data)):
            self.key_label.append(item.pop('key_label', ''))
            self.container.append(item)


    def astype(self, T):
        """ No-op for numpy array compatibility """
        return self


    def __array__(self, *args, **kwargs):
        """ Return a one-element numpy object array containing this Sample """
        array = np.empty(1, dtype=object)
        array[0] = self
        return array


    def __getitem__(self, idx: int | str) -> xr.Dataset:
        """ Retrieve a Datafile window as an xarray Dataset """
        if isinstance(idx, str):
            idx = self.index(idx)
        elif idx < 0:
            idx += len(self)

        if not 0 <= idx < len(self):
            raise IndexError(f'Sample index out of range: {idx}')
        if idx not in self.cache:
            self.cache[idx] = self._load(idx)
        return self.cache[idx]


    def __iter__(self) -> Iterator[xr.Dataset]:
        """ Iterate over the Datafile windows """
        yield from map(self.__getitem__, range(len(self)))


    def __len__(self) -> int:
        """ Return the number of Datafile windows """
        return len(self.container)


    def _load(self, idx: int) -> xr.Dataset:
        """ Instantiate an xarray object from one raw data dictionary """
        return xr.DataArray(**self.container[idx]).to_dataset('features')


    def index(self, feature: str) -> int:
        """ Find the index of the first item containing a feature """
        for i,item in enumerate(self.container):
            if feature in item['coords']['features']:
                return i
        raise ValueError(f'Feature "{feature}" not found in: {self.features}')


    @property
    def nbytes(self) -> int:
        """ Get the total number of bytes used by the xarray windows """
        return sum(data.nbytes for data in self)


    @property
    def dims(self) -> list[str]:
        """ All dimensions, in the same order as the original data dims """
        dims = []
        for item in self.container:
            for dim in item['coords']:
                if dim not in dims:
                    dims.append(dim)
        return dims


    @property
    def coords(self) -> dict[str, list]:
        """ Coords for all items; {dim: [item0_dim, item1_dim, ...]} """
        coords = {}
        for item in self.container:
            for dim,values in item['coords'].items():
                coords.setdefault(dim, []).append(values)
        return coords


    @property
    def coords_avg(self) -> dict[str, float]:
        """ Average numeric coordinates across all Datafile windows """
        averages = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            for dim in self.dims:
                values = []
                for item in self.container:
                    if dim not in item['coords']:
                        continue

                    coord = np.asarray(item['coords'][dim])
                    if np.issubdtype(coord.dtype, np.number):
                        values.append(coord.mean())

                if values:
                    averages[dim] = np.nanmean(values)
        return averages


    def get_feature_coords(self, feature: str) -> dict[str, np.ndarray]:
        """ Return the coordinate arrays for a specific feature """
        coords = self.container[self.index(feature)]['coords']
        return {k:v for k,v in coords.items() if k != 'features'}


    @property
    def data(self) -> dict[str, list | dict]:
        """ Data for all items; {feature: {item_index: value, ...}} """
        features = {}
        for i,item in enumerate(self.container):
            names = item['coords']['features']
            for name,value in zip(names, item['data']):
                features.setdefault(name, {})[i] = value
        return self.coords | features


    @property
    def data_groups(self) -> list[dict]:
        """ Data grouped by Datafile; [{feature: value, coord: value, ...}] """
        groups = []
        for item in self.container:
            features = dict(zip(item['coords']['features'], item['data']))
            groups.append(features | item['coords'])
        return groups


    @property
    def features(self) -> list:
        """ All available features in physical Datafile order """
        return [feature for item in self.container
                for feature in item['coords']['features']]


    @staticmethod
    def _make_nan(key: str) -> np.ndarray:
        """ Create a NaN or NaT placeholder for a missing coordinate """
        if key == 'datetime':
            return np.array(['NaT'], dtype='datetime64[ns]')
        return np.array([np.nan], dtype='float32')


    def _get_index(self, key: object, feature: str|None = None) -> int:
        """ Resolve a Datafile index or escaped key label """
        key = str(key).replace('$', '/')
        if key in self.key_label:

            # Multiple sources can have the same label, but differing features
            if feature is not None:
                for i, label in enumerate(self.key_label):
                    if label == key:
                        names = self.container[i]['coords']['features']
                        if feature in names:
                            return i

            # Default behavior is to return the first source with this label
            return self.key_label.index(key)

        try:
            index = int(key)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f'Unknown key/format: {key=} {self.key_label=}') from e

        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise ValueError(f'Unknown key/format: {key=} {self.key_label=}')
        return index


    def _get_feature(self, feature: str, index: int) -> np.ndarray:
        """ Return a feature from one specific Datafile window """
        item = self.container[index]
        for i,name in enumerate(item['coords']['features']):
            if name == feature:
                return item['data'][i]
        raise ValueError(
            f'Feature "{feature}" not found in Datafile[{index}]: '
            f'{list(item["coords"]["features"])}')


    def _get_coord(self, coord: str, index: int) -> np.ndarray:
        """ Return a coordinate or a typed missing-value placeholder """
        coords = self.container[index]['coords']
        if coord in coords:
            return coords[coord]
        return self._make_nan(coord)

    
    def _get_value(self, feature: str, index: int) -> np.ndarray:
        """ Return a data feature or coordinate from one Datafile """
        item = self.container[index]

        if feature in item['coords']['features']:
            return self._get_feature(feature, index)
        if feature in item['coords'] or feature in self.dims:
            return self._get_coord(feature, index)
        raise ValueError(
            f'Value "{feature}" not found in Datafile[{index}]')

        
    def _select_distance(self, selector: str) -> dict[str, np.ndarray]:
        """ Return pairwise coordinate differences between two Datafiles """
        if selector.count(':') != 2:
            raise ValueError(f'{selector=} does not match the expected format '
                             '"distance:{key1}:{key2}"')

        key1,key2 = selector.split(':')[1:]
        index1 = self._get_index(key1)
        index2 = self._get_index(key2)
        if index1 == index2:
            raise ValueError(f'{selector=}: {index1=} == {index2=}')

        def difference(dim: str) -> np.ndarray:
            """ Calculate one coordinate's pairwise difference matrix """
            coord1 = np.atleast_1d(self._get_coord(dim, index1))
            coord2 = np.atleast_1d(self._get_coord(dim, index2))
            return coord1[:, None] - coord2[None]

        return {dim: difference(dim) for dim in self.dims
                if dim != 'features'}


    def _select_labeled(self, selector: str) -> np.ndarray:
        """ Select a value using the ``label__feature`` syntax """
        label, feature = selector.split('__', 1)
        return self._get_value(feature, self._get_index(label, feature))


    def _select_indexed(self, selector: str) -> np.ndarray:
        """ Select a value using the ``feature@source`` syntax """
        feature, key = selector.rsplit('@', 1)
        if not feature or not key:
            raise ValueError(f'Invalid feature selector: {selector=}')
        return self._get_value(feature, self._get_index(key, feature))

    
    def _select(self, selector: object):
        """ Resolve one feature-selector expression """
        if not isinstance(selector, str):
            return self.data[selector][0]

        if selector.startswith('distance:'):
            return self._select_distance(selector)
        if '__' in selector:
            return self._select_labeled(selector)
        if '@' in selector:
            return self._select_indexed(selector)

        # Data variables take precedence over coordinate names
        try:
            return self._get_feature(selector, self.index(selector))
        except ValueError:
            for i,item in enumerate(self.container):
                if selector in item['coords']:
                    return self._get_coord(selector, i)
            raise


    def to_list(self, features: list | None = None) -> list:
        """ Extract the requested features into a list """
        # No explicit selection means retaining each physical feature occurrence
        if features is None:
            return [value for item in self.container for value in item['data']]
        return [self._select(feature) for feature in features]


    def to_array(self, features: list | None = None) -> np.ndarray:
        """ Extract the requested features into an array """
        return np.array(self.to_list(features))


    def to_dict(self, features: list | None = None) -> dict:
        """ Extract the requested features into a dictionary """
        if features is None:
            return dict(zip(self.features, self.to_list()))
        return dict(zip(features, self.to_list(features)))
