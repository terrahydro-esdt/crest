import dask.array as da 
import numpy as np 
import pytest
import dask 
import time 

from crest.src.data import Batcher
    

@pytest.mark.parametrize('batch_size', (1, 10, 100))
@pytest.mark.parametrize('shuffle', (True, False))
def test_batcher(batch_size, shuffle):
    data    = np.arange(1000)
    samples = list(da.from_array(data.copy(), chunks=50).to_delayed())
    batcher = Batcher(samples, batch_size, shuffle=shuffle)
    batches = [b for batch in batcher for b in list(batch)]

    if shuffle: batches = sorted(batches)
    assert(batches == list(data))


@pytest.mark.parametrize('batch_size', (5, 20))
@pytest.mark.parametrize('sleep', (0.5,))
@pytest.mark.parametrize('size',  (50, 300))
def test_speed(batch_size, sleep, size):

    @dask.delayed
    def generate(i):
        time.sleep(sleep)
        return da.from_array(np.arange(size) + (i * size)).rechunk(100)

    # Create a dataset with 5 blocks that has 100 samples / chunk
    dataset = list(map(generate, range(20)))
    batcher = Batcher(dataset, batch_size, shuffle=True)

    # Calculate the theoretical maximum number of batches / second
    t_max = ((batcher.workers * size) / batch_size) / sleep
    times = [time.time()] + [time.time() for _ in batcher]
    speed = 1/np.mean([e-s for s,e in zip(times[:-1], times[1:])])

    # Speed should be within ~10% of the theoretical maximum
    assert(speed > (t_max * 0.8)), f'{speed:.2f} vs {t_max:.2f}'