from collections.abc import Collection
import xarray as xr
import numpy as np 

from ...base.BaseSet import BaseSet
from ...utils.partial_product import partial_product
from .Sample import Sample


class SampleSet(BaseSet):
    """ Wraps the data windows to allow interfacing with dask.

    Notes
    -----
    Slightly different than a normal BaseSet object, as the container
    contents are actually the raw data windows rather than Sample wrappers.
    This is done in order to allow creating the product of data windows at
    the last minute - as the memory footprint of several lists is much smaller
    than the cartesian product of those lists.
    
    The raw data windows used to initialize this object can be understood as:
        [ 
          [Datafile_1 window_1, Datafile_1 window_2, ...], # Datafile_1 windows
          [Datafile_2 window_1, Datafile_2 window_2, ...], # Datafile_2 windows
          ...
        ]
    where a Datafile window is the window of data extracted from a given
    Datafile, using the given window_depth definitions (e.g. 3 x 3 x 2 
    window [latitude x longitude x time]).

    In other words, the passed in `windows` parameter is a collection of 
    Datafile window collections, where there is exactly one collection of 
    windows per Datafile (i.e. len(windows) == len(Dataset)). The number 
    of windows inside each Datafile window collection depends on the number 
    of valid windows found during the search procedure.

    This class shouldn't be used externally, and only exists to interface
    properly with dask and allow a lazy cartesian product of data windows. 

    Parameters
    ----------
    windows   : Collection[Collection[xr.Dataset]]
        The collection of data window collections, where 
        len(windows) == len(Dataset).
    singleton : bool 
        Determines whether SampleSet.values will return the expanded list 
        of Samples, or itself. Dask does not expand values if there is only
        a single partition, and so when that is the case we need to force the
        expansion of Samples during the .values call. This ensures that when 
        compute is called on the final array, it will result in an array of
        Samples rather than a single SampleSet object.

    """
    def __init__(self, 
        windows   : Collection[Collection],
        singleton : bool = False,
        dtype = object,
    ):
        self.container = windows
        self.singleton = singleton
        self.ele_dtype = dtype
        self.n_samples = [np.prod(list(map(len, w))) for w in windows]


    @property
    def values(self):
        """ Dask expects the return value of dd.from_map to be a dataframe
        object which returns an array via a .values attribute. We circumvent
        this restriction by just returning ourself via .values, and performing
        the lazy window lookup when actually fetching via __getitem__ """
        return self[:] if self.singleton else self


    @property
    def shape(self):
        """ Required to trigger dask appropriately """
        return (len(self),)
    

    def __array__(self, *args, **kwargs): 
        """ Dask attempts to pass the SampleSet into a numpy array """
        return self[:]


    def __len__(self):
        """ Length of the cartesian product """
        return np.sum(self.n_samples)

    
    def __getitem__(self, idx) -> Sample | np.ndarray:#[Sample]:
        """Performs the lazy cartesian product over data windows.
        
        Notes
        -----
        Any slicing convention can be used to retrieve Samples from this class;
        e.g. SampleSet[:3], SampleSet[1:5:2], SampleSet[slice(4, -2)], etc.

        The core of this method performs a partial cartesian product over the
        container windows, where only the requested data windows are actually 
        generated. This allows constant-time lookups on the full Sample array,
        regardless of the location to be retrieved. 

        """
        # We only retrieve on the first dimension, which is the list of Samples
        index = idx[0] if hasattr(idx, '__getitem__') else idx
        multi = hasattr(index, 'start')
        
        # If we're not retrieving multiple samples, wrap the index in a slice
        if not multi: index = slice(index, index+1)
        
        # Calculate the partial cartesian product and wrap each set with Sample
        samples = [Sample(m, self.ele_dtype) for m in self._matches(index)]

        # Return multiple Samples in a numpy array to appease dask
        if multi or isinstance(index.start, (list, np.ndarray)):
            results = np.empty(len(samples), dtype=object)
            results[:] = samples
            return results

        # Otherwise just return the single retrieved Sample object
        return samples[0]


    def _matches(self, idx):
        """ Extract matches from the nested container """
        start, stop, step = idx.start, idx.stop, idx.step 
        start = start or 0
        stop  = stop  or len(self)

        for size, match in zip(self.n_samples, self.container):
            if start < size:
                yield from partial_product(match, start, stop, step)

            start = max(0, start-size)
            stop -= size

            if stop < 0: break
