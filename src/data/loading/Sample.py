from collections.abc import Collection, Iterator
import xarray as xr 
import numpy as np 

from crest.src.base import BaseAbstract 


class Sample(BaseAbstract):
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
    data : Collection[xr.Dataset]
        The collection of data windows, with exactly one per Datafile.

    """
    def __init__(self, data: Collection[xr.Dataset]):  
        self.data = data 


    def __getitem__(self, idx) -> xr.Dataset | Collection[xr.Dataset]:
        """ Retrieve a subset of the full collection """
        return self.data[idx]
    

    def __iter__(self) -> Iterator[xr.Dataset]:
        """ Iterate over the collection """
        yield from self.data 


    def __len__(self) -> int:
        """ Return the number of items in the collection """
        return len(self.data)


    @property
    def nbytes(self) -> int:
        """ Get the total number of bytes used by the data """
        return sum(d.nbytes for d in self.data)
