from dask.delayed import Delayed
from functools import cached_property

import numpy as np
import dask.array as da
import dask

from crest.data.loading.Sample import Sample


class StructuredDataset:
    """Constrained alternative to the the more general Dataset class.
    
    Notes
    -----
    A StructuredDataset can provide data much faster to the Batcher than the 
    more generalized Dataset class, but has two important constraints:
      - must be able to fit in memory
      - all features are aligned along the first dimension (i.e. n_samples)
    In general, data which can be passed as-is to a machine learning model
    would adhere to these constraints (e.g. MNIST).

    Parameters
    ----------
    *data  : np.ndarray
        Any number of numpy arrays, which should each be shaped 
        (n_samples, ...). These arrays must all have the 
        same leading dimension (i.e. the same number of samples), as the
        first dimension will be used to group the arrays into samples.
        Each array given will be a separate feature in the Sample 
        objects that are created.
    labels : list[str], optional 
        A list of strings can be given if specific labels are desired 
        for the features. If no labels are given, features will be labeled
        as the string index in the order in which they are given; i.e. 
        {'0': <first array>, '1': <second array>, ...}. 
    chunks : int, optional
        Chunk size of the lazy dask Sample array. Since by definition the
        data fits in memory, in most cases the chunk size should be -1 
        (i.e. all the data is in a single block).
    blocks : int, optional
        How many blocks the lazy dask Sample array should be split into
        when using generate_samples(compute=False). Similar to chunks,
        in most cases this should be 1 so that all data is contained in
        a single block. 

    """

    def __init__(self, 
        *data  : np.ndarray, 
        labels : list[str] = [],
        chunks : int = -1,
        blocks : int = 1,
    ):
        self.data   = data 
        self.labels = labels or list(map(str, range(len(data))))
        self.chunks = chunks
        self.blocks = blocks

        self.features = n_features = len(data)
        self.samples  = n_samples  = len(data[0])

        # Verify all data have the same first dimension length
        assert(all(n == n_samples for n in map(len, data)))
        assert(blocks <= n_samples)
        assert(len(self.labels) == n_features)


    def __getstate__(self):
        """ Remove _delayed_blocks since lambdas are not pickleable """
        state = dict(self.__dict__)
        state.pop('_delayed_blocks', None)
        return state


    @cached_property
    def _sample_dict(self) -> list[dict]:
        """ One dictionary per data array to instantiate Sample objects """
        dims   = [list(map(str, range(d.ndim-1))) for d in self.data]
        coords = [map(np.arange, d.shape[1:]) for d in self.data]
        return [{
            'dims'   : dim + ['features'],
            'coords' : dict(zip(dim, coord)) | {'features': [label]},
        } for label, dim, coord in zip(self.labels, dims, coords)]


    def _to_sample(self, *data: np.ndarray) -> Sample:
        """ Transform data elements into Sample objects """ 
        to_dict = lambda arr, sd: {'data': [arr]} | sd
        return Sample( list(map(to_dict, data, self._sample_dict)) )


    @cached_property
    def _lazy_array(self) -> da.Array:
        """ Transform data into a lazy dask array of Samples """
        sample  = np.frompyfunc(self._to_sample, nin=self.features, nout=1)
        data    = np.empty((self.features, self.samples), dtype=object)
        data[:] = list(map(list, self.data))
        return da.from_array(sample(*data), chunks=self.chunks, name=False, inline_array=True)


    @cached_property
    def _delayed_blocks(self) -> list[Delayed]:
        blksize = self.samples // self.blocks
        indices = np.arange(0, self.samples, blksize)
        slices  = map(slice, indices, indices + blksize)
        blocks  = map(self._lazy_array.__getitem__, slices)
        delay   = lambda block: dask.delayed(lambda b=block: b)()
        return list(map(delay, blocks))


    def generate_samples(self, *args, compute=False, **kwargs):
        """ Mimics the returned values of Dataset.generate_samples """
        return self._lazy_array if compute else self._delayed_blocks
        if compute: return self._lazy_array

        # Blocks need to be delayed objects which produce lazy Sample arrays,
        # but there doesn't seem to be a way to prevent dask from computing the
        # inner lazy Sample array when computing the outer delayed block object
        # without using lambda functions.
        #
        # Lambda functions can't be pickled however, and so the final list of 
        # blocks can't be cached like the lazy Sample array. Fortunately, the 
        # process of creating delayed blocks is very fast unless there are many
        # that need to be created (which there shouldn't be for data that fits
        # in memory), and so the time lost from doing it repeatedly is negligible
        blksize = self.samples // self.blocks
        indices = np.arange(0, self.samples, blksize)
        slices  = map(slice, indices, indices + blksize)
        blocks  = map(self._lazy_array.__getitem__, slices)
        delay   = lambda block: dask.delayed(lambda b=block: b)()
        return list(map(delay, blocks))
