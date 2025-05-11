import dask.array as da 
import xarray as xr
import numpy as np 
import pytest
import string
import dask 
import time 

from crest.data.loading import Dataset, Datafile, Sample
from crest.data import Batcher


@pytest.mark.parametrize('batch_size', (1, 10, 100))
@pytest.mark.parametrize('shuffle', (True, False))
def test_batcher(batch_size, shuffle):
    """ Ensure all samples are returned, and in order if shuffle=False """
    data    = np.arange(1000)
    samples = list(da.from_array(data.copy(), chunks=50).to_delayed())
    batcher = Batcher(samples, batch_size, shuffle=shuffle, workers=0)
    batches = [b for batch in batcher for b in list(batch)]

    if shuffle: batches = sorted(batches)
    assert(batches == list(data))


@pytest.mark.speed
@pytest.mark.parametrize('batch_size', (16,))
@pytest.mark.parametrize('sleep', (0.5,))
@pytest.mark.parametrize('size',  (100,))
def test_theoretical_max(batch_size, sleep, size):
    """ Ensure batches can be delivered close to the theoretical max speed """
    @dask.delayed
    def generate(i):
        time.sleep(sleep)
        return da.from_array(np.arange(size) + (i * size), chunks=50)

    # Create a dataset with 10 blocks that has 100 samples / chunk
    dataset = list(map(generate, range(15)))
    batcher = Batcher(dataset, batch_size, shuffle=True, workers=0)

    # Calculate the theoretical maximum number of batches / second
    t_max = ((batcher.threads * size) / batch_size) / sleep
    times = [time.time()] + [time.time() for _ in batcher]
    speed = 1/np.mean([e-s for s,e in zip(times[:-1], times[1:])])

    # Speed should be within ~10% of the theoretical maximum
    assert(speed > (t_max * 0.8)), f'{speed:.2f} vs {t_max:.2f}'


@pytest.mark.parametrize('batch_size', (2,))
@pytest.mark.parametrize('n_features', (5,))
@pytest.mark.parametrize('shape',      ((10,2),))
def test_features(batch_size, n_features, shape):
    """ Ensure batches contain extracted feature dicts with correct shapes """
    def create_dataset(*size, n_features=2, **kwargs):
        data   = np.arange(int(np.prod(size))).reshape(size)
        coords = dict(zip(string.ascii_lowercase, map(np.arange, size)))
        x_data = {str(i): xr.DataArray(data, coords) for i in range(n_features)}
        return Dataset([Datafile(xr.Dataset(x_data), **kwargs)])

    # First test extracting Sample objects
    depth   = {'a':(0, shape[0]-1)}
    dataset = create_dataset(*shape, n_features=n_features, window_depth=depth)
    with Batcher(dataset, batch_size=batch_size, workers=0) as batches:
        batch = next(batches)

    # Ensure batch is the correct size, and contains Sample objects
    assert(len(batch) == batch_size), f'{len(batch)} vs {batch_size}'
    assert(isinstance(batch[0], Sample)), batch[0]

    # Ensure Sample dict is correct
    d = batch[0].to_dict()
    assert(len(d) == n_features), f'{len(d)} vs {n_features}'
    shapes = [v.shape for v in d.values()]
    assert(all(s == (shape[0], 1) for s in shapes)), f'{shapes} vs {shape}'

    # Test extracting feature dicts this time
    with Batcher(dataset, batch_size=batch_size, workers=0, features=[]) as batches:
        batch = next(batches)

    assert(isinstance(batch, dict)), type(batch)
    assert(len(batch) == n_features), f'{len(batch)} vs {n_features}'
    shapes = [v.shape for v in batch.values()]
    target = (batch_size, shape[0], 1)
    assert(all(s == target for s in shapes)), f'{shapes} vs {target}'

    # Last, test extracting nested feature dicts
    features = [['0'], [['1', '2'], ['3']], ['4']]
    with Batcher(dataset, batch_size=batch_size, workers=0, features=features) as batches:
        batch = next(batches)

    def recurse(features, batch):
        """ Recursively check nested features """
        assert(len(features) == len(batch)), f'{len(batch)} vs {len(features)}'
        if isinstance(features[0], list):
            assert(not isinstance(batch, dict)), [features, batch]    
            return list(map(recurse, features, batch))
        assert(isinstance(batch, dict)), [features, batch]
        assert(all(f in batch for f in features)), f'{features} vs {list(batch.keys())}'
        shapes = [batch[f].shape for f in features]
        assert(all(s == target for s in shapes)), f'{shapes} vs {target}'
    recurse(features, batch)

    
def sync_generate(i):
    time.sleep(i % 4)
    return da.from_array(np.array([i]), chunks=-1)

def test_block_sync():
    """ Ensure batches are delivered in order when workers are synchronized """
    dataset = list(map(dask.delayed(sync_generate), range(6)))
    batcher = Batcher(dataset, 1, shuffle=False, workers=3, block_sync=True)
    batches = [b for batch in batcher for b in list(batch)]
    assert(set(batches[:3]) == {0,1,2}), batches
    assert(set(batches[3:]) == {3,4,5}), batches

