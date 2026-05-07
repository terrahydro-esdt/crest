from collections.abc import Collection, Iterator
from collections import defaultdict as dd
from functools import cached_property
from tlz import merge_with, dissoc
import re

import xarray as xr 
import numpy as np 
import warnings 

from crest.base import BaseAbstract 


# class DatafileSample:
#     def __init__(self, requested_features: list[str] = [], **xr_data):
#         self.features = requested_features
#         self._xr_data = xr_data

#     @cached_property
#     def data(self) -> xr.Dataset:
#         """ Instantiate the xarray object from the raw data dict """
#         return xr.DataArray(**self._xr_data).to_dataset('features')



# Inheriting from BaseAbstract nearly triples the time for loading batches
# e.g. ~190 Batches/sec -> ~70 Batches/sec
class Sample:#(BaseAbstract):
    """ Class which holds all Datafile windows for a single location. 

    Notes
    -----
    When a valid window is found across all Datafiles, a Sample is
    created which holds the xr.Dataset object for each Datafile.

    These xr.Dataset objects contain the window which was defined
    for each Datafile, respectively, when the search was performed 
    (e.g. 3 x 3 x 2 window [latitude x longitude x time]).

    Each xr.Dataset object contains all variables from the respective
    Datafile object it represents, as well as the coordinates from 
    which those variables were retrieved. 

    Parameters
    ----------
    data : Collection[dict]
        The collection of data windows, with exactly one per Datafile.
        The dictionary elements should each contain all information 
        necessary to initialize the respective xarray.DataArray object
        (i.e. data, coords, and dims, with one dim named 'features').

    """
    def __init__(self, data: Collection[dict], dtype=object):
        self._features = merge_with(list, [dict.fromkeys(d.get('requested_features', []), i) for i,d in enumerate(data)])
        self.key_label = [d.get('key_label', '') for d in data]
        self.container = [dissoc(d, 'requested_features', 'key_label') for d in data]
        self.dtype = dtype
        self.cache = {}


    def astype(self, T):
        return self 


    def __array__(self, *args, **kwargs): 
        a = np.empty(1, dtype=object)
        a[0] = self
        return a


    def __getitem__(self, idx: int | str) -> xr.Dataset:
        """ Retrieve a subset of the full collection """
        if isinstance(idx, str): idx = self.index(idx)
        return self.cache.setdefault(idx, self._load(idx))


    def __iter__(self) -> Iterator[xr.Dataset]:
        """ Iterate over the collection """
        yield from map(self.__getitem__, range(len(self)))


    def __len__(self) -> int:
        """ Return the number of items in the collection """
        return len(self.container)


    def _load(self, idx: int) -> xr.Dataset:
        """ Instantiate the xarray object from the raw data dict """
        return xr.DataArray(**self.container[idx]).to_dataset('features')


    def index(self, feature: str) -> int:
        """ Find the index of the (first) item containing the feature """
        for i, item in enumerate(self.container):
            if feature in item['coords']['features']:
                return i
        raise Exception(f'Feature "{feature}" not found in: {self.features}')


    @property
    def nbytes(self) -> int:
        """ Get the total number of bytes used by the data """
        return sum(d.nbytes for d in self)


    @property
    def dims(self) -> list:
        """ All dimensions, in the same order as the original data dims """
        union = [o for item in self.container for o in item['coords']]
        return sorted(set(union), key=union.index)


    @property
    def coords(self) -> dict[str, list]:
        """ Coords for all items; {dim: [item0_dim, item1_dim, ...]} """
        return merge_with(list, *[item['coords'] for item in self.container])


    @property
    def coords_avg(self) -> dict[str, float]:
        """ Coords for all items; {dim: [item0_dim, item1_dim, ...].mean()} """
        coords = [item['coords'] for item in self.container]
        is_num = lambda v: np.issubdtype(v.dtype, np.number)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return {dim: np.nanmean([c[dim].mean() for c in coords if dim in c and is_num(c[dim])]) for dim in self.dims}


    def get_feature_coords(self, feature: str):
        """ Return the coordinate arrays for a specific feature """
        return dissoc(self.container[self.index(feature)]['coords'],'features')


    @property
    def data(self) -> dict[str, list]:
        """ Data for all items; {feature: [item0_feature, ...]} """
        to_dict = lambda i,d: dict(zip(d['coords']['features'], [(i,v) for v in d['data']]))
        i_dicts = [to_dict(i, item) for i, item in enumerate(self.container)]
        return self.coords | merge_with(dict, *i_dicts)


    @property
    def data_groups(self) -> list[dict]:
        """ Data grouped by datafile; [{feature: [df0_feature, ...]}] """
        to_dict = lambda d: dict(zip(d['coords']['features'], d['data']))
        return [to_dict(item) | item['coords'] for item in self.container]


    @property
    def features(self) -> list:
        """ All available features """
        return [f for d in self.container for f in d['coords']['features']]
    

    def to_list(self, features: list | None = None) -> list:
        """ Extract the requested features into a list """
        # Load data / groups lazily
        data = None
        grps = None
        vals = []

        def get_index(i):
            i = str(i).replace('$', '/')
            if i in self.key_label:
                return self.key_label.index(i)
            try: 
                return int(i)
            except:
                raise Exception(f'Unknown format: {i=} {self.key_label=}')
                
        for feature_index in (features or self.features):
            if isinstance(feature_index, str):
                # Handle feature differences (i.e. A-B)
                # if feature_index.startswith('difference:'):
                #     if data is None: data = self.data
                #     _, d1, d2 = feature_index.split(':')
                #     f1, i1, *_ = d1.split('@')
                #     f2, i2, *_ = d2.split('@')
                #     assert(i1 != i2), f'{feature_index=} {i1=} {i2=}'
                #     v1 = data[f1][get_index(i1)]
                #     v2 = data[f2][get_index(i2)]
                #     diff = v1 - v2
                #     diff[diff != 0] = abs(diff[diff != 0]) ** 0.5 * np.sign(diff[diff != 0])
                #     vals.append(diff)

                if '>>' in feature_index:
                    if data is None: data = self.data
                    label,feature = feature_index.split('>>',1)
                    # Labels are ignored if not found
                    if label not in self.key_label:
                        label = self._features.get(feature, [0])[0]
                    vals.append(data[feature][get_index(label)])
                    
                elif '@' in feature_index:
                    if data is None: data = self.data
                    if feature_index not in data:
                        feature, *index = (feature_index+'@0').split('@')
                        vals.append(data[feature][get_index(index[0])])
                    else: vals.append(list(data[feature_index].values())[0])
                        
                else:
                    if feature_index not in self._features:
                        feature, *index = (feature_index+'@0').split('@')

                        if feature in self._features:
                            if grps is None: grps = self.data_groups
                            f_idx = self._features[feature][int(index[0])]
                            vals.append(grps[f_idx][feature])
                        else:
                            if data is None: data = self.data
                            if feature_index not in data:
                                vals.append(data[feature][int(index[0])])
                            elif isinstance(data[feature_index], dict): 
                                vals.append(list(data[feature_index].values())[0])
                            else: vals.append(data[feature_index][0])
                    else: 
                        if grps is None: grps = self.data_groups
                        f_idx = self._features[feature_index][0]
                        vals.append(grps[f_idx][feature_index])
            else: 
                if data is None: data = self.data
                vals.append(data[feature_index][0])



            # # Allow selecting feature from specific container item when there
            # # are duplicates; e.g. 'latitude@2' is latitude from container[2]
            # if isinstance(feature_index, str) and feature_index not in data:
            #     feature, *index = (feature_index+'@0').split('@')
            #     vals.append(data[feature][int(index[0])])
            # else: vals.append(data[feature_index][0])
        # BaseAbstract.interactive()
        assert(len(vals) == len(features or vals)), \
            f'Missing / duplicate features: {len(vals)} vs {len(features)}\n'+\
            f'Expected: {features}'
        return vals 


    def to_array(self, features: list | None = None) -> np.ndarray:
        """ Extract the requested features into an array """
        return np.array(self.to_list(features))


    def to_dict(self, features: list | None = None) -> dict:
        """ Extract the requested features into a dictionary """
        if features is None: features = self.features
        return dict(zip(features, self.to_list(features)))
