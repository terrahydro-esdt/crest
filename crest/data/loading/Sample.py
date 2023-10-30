from collections.abc import Collection, Iterator
import xarray as xr 
import numpy as np 

from crest.base import BaseAbstract 


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
        self._data = data
        self._dict = [dict(zip(d['coords']['features'], d['data'])) for d in data]
        self.dtype = dtype
        self.cache = {}


    def astype(self, T):
        return self 


    def __array__(self, *args, **kwargs): 
        a = np.empty(1, dtype=object)
        a[0] = self
        return a


    def __getitem__(self, idx: int) -> xr.Dataset:
        """ Retrieve a subset of the full collection """
        return self.cache.setdefault(idx, self._load(idx))


    def __iter__(self) -> Iterator[xr.Dataset]:
        """ Iterate over the collection """
        yield from map(self.__getitem__, range(len(self)))


    def __len__(self) -> int:
        """ Return the number of items in the collection """
        return len(self._data)


    def _load(self, idx) -> xr.Dataset:
        """ Instantiate the xarray object from the raw data dict """
        # Annoying process to get xarray to order coords/features properly
        # Necessary in order to remove (1,) dimensions from the data in block.py,
        # but keep them in the xarray object. Need to rewrite this to be more robust
        data  = dict(self._data[idx])
        order = data['order']
        attrs = data.get('attrs', {})
        dims  = data['dims'][1:]
        coord = dict(data['coords'])
        coord = {k:coord[k] for k in order if k in coord}
        data  = [(dims, d) for d in data['data']]
        feat  = coord.pop('features')
        return xr.Dataset(dict(zip(feat, data)), coord, attrs).transpose(*order[1:])[list(order[1:])+list(feat)]
        # return xr.DataArray(**self._data[idx]).to_dataset('features')


    @property
    def nbytes(self) -> int:
        """ Get the total number of bytes used by the data """
        return sum(d.nbytes for d in self)


    @property
    def features(self) -> list:
        """ All available features """
        return [f for d in self._dict for f in d]
    

    def to_list(self, features: list | None = None) -> list:
        """ Extract the requested features into a list """
        data = [d[f] for f in (features or d) for d in self._dict if f in d]
        assert(len(data) == len(features or data)), \
            f'Missing / duplicate features: {len(features)} vs {len(data)}'
        return data 


    def to_array(self, features: list | None = None) -> np.ndarray:
        """ Extract the requested features into an array """
        return np.array(self.to_list(features))


    def to_dict(self, features: list | None = None) -> dict:
        """ Extract the requested features into a dictionary """
        if features is None: features = [k for d in self._dict for k in d]
        return dict(zip(features, self.to_list(features)))