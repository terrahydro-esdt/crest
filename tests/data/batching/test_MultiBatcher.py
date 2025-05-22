from itertools import starmap
import xarray as xr
import numpy as np 
import logging
import pytest
import string

from crest.data.batching import MultiBatcher
from crest.data.loading import Dataset, Datafile


@pytest.mark.parametrize('batch_size', (1,))
@pytest.mark.parametrize('n_features', (1,))
@pytest.mark.parametrize('n_workers',  (0,1))
@pytest.mark.parametrize('shape',      ((10,2),))
def test_multibatcher(batch_size, n_features, n_workers, shape):
    """ Ensure batches contain extracted feature dicts with correct shapes """
    def create_dataarray(shape):
        data = np.arange(int(np.prod(shape))).reshape(shape).astype(float)
        coords = dict(zip(string.ascii_lowercase, map(np.arange, shape)))
        return xr.DataArray(data, coords)

    # Insert NaNs into the arrays to change which valid samples are created
    xr_arrays = [create_dataarray(shape) for _ in range(4)]
    xr_arrays[0].data[ :3, 0] = np.nan
    xr_arrays[1].data[3:6, 1] = np.nan
    xr_arrays[2].data[6:] = np.nan
    xr_arrays[3].data[:2] = np.nan
    xr_arrays[3].data[ 3] = np.nan
    xr_arrays[3].data[5:] = np.nan

    # Print the data we're using for debugging purposes
    stacked = np.stack([d.data.T for d in xr_arrays], axis=0)
    valmask = np.where(np.isfinite(stacked).all(0), 1, np.nan)
    print(f'\nAll data:\n{stacked}')
    print(f'\nAll valid:\n{stacked[0] * valmask}')

    # Create datasets with globally unique feature names
    to_dict = lambda i, data: {f'{i}_{f}': data for f in range(n_features)}
    xr_sets = [xr.Dataset(d) for d in starmap(to_dict, enumerate(xr_arrays))]
    dataset = Dataset([Datafile(data) for data in xr_sets])
    
    features = list(dataset.data.features.values.tolist())
    for fs in features:
        if 'valid_mask' in fs: 
            fs.remove('valid_mask')
    print(f'{features=}')
    # Get all of the batches using the MultiBatcher with drop_datafiles
    batcher_kwargs = {
        'drop_datafiles'    : [[1, 2], [3]],
        'directed_sampling' : False,

        'batch_size' : batch_size, 
        'workers'    : n_workers, 
        'features'   : features, 
        'numblocks'  : (2, 1), 
        'shuffle'    : False,
        'log_level'  : logging.DEBUG, 
    }
    with MultiBatcher(dataset, **batcher_kwargs) as batcher:
        batches = list(batcher)
    expectation = [[5, 10], [np.nan, 10], [np.nan, 10], [5, np.nan]]

    # Verify the result matches expectations (single batch with 4 dicts)
    # If 0 batches are found in the second test (which uses n_workers > 0), it
    #   is likely due to a worker grabbing block #1 with configuration #0. For
    #   the test to be successful, the two blocks need to be paired with the
    #   two respective configurations (0 with 0, 1 with 1).
    assert(len(batches) == 1), len(batches)
    assert(len(batches[0]) == 4), len(batches[0])
    for i, (result, expected) in enumerate(zip(batches[0], expectation)):
        assert(isinstance(result, dict)), type(result)
        assert(len(result) == 1), result
        print(i, result, expected, batches)
        np.testing.assert_equal(list(result[f'{i}_0'].flatten()), expected)