import dask.array as da 
import numpy as np 
import pytest

from crest.src.data import Batcher
    

@pytest.mark.parametrize('batch_size', (1, 10, 100))
@pytest.mark.parametrize('shuffle', (True, False))
def test_batcher(batch_size, shuffle):
    data    = np.arange(1000)
    samples = da.from_array(data, chunks=50)
    batcher = Batcher(samples, batch_size, shuffle=shuffle)
    batches = [b for batch in batcher for b in list(batch)]

    if shuffle: batches = sorted(batches)
    assert(batches == list(data))
