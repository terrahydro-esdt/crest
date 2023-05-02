from functools import partial
import pytest
import xarray as xr 
import numpy as np 

from crest.src.data.loading import Dataset, Datafile
from crest.src.utils import synthetic_data
from .Dataset_config import configs 


def check_outputs(config, expected, update_data=lambda x: x):
    """ Run the given config, and check if outputs match expectations """
    config = config.copy()
    depths = config.pop('depth')
    
    data = update_data( synthetic_data(**config) )
    dfs  = [Datafile(d, window_depth=wd) for d, wd in zip(data, depths)]

    dataset = Dataset(dfs)
    outputs = dataset.generate_samples()

    for output, expect in zip(outputs.compute(), expected):
        for out, exp in zip(output, map(tuple, expect)):
            get_mid = lambda v: v[len(v) // 2].values.item()
            get_out = lambda d: round(get_mid(out[d]) / out.resolution[d], 6)
            center  = tuple(map(get_out, out.dims))
            equals  = partial(np.array_equal, equal_nan=True)
            assert(equals(center, exp)), [center, exp]
        assert(len(output) == len(expect))
    assert(len(outputs) == len(expected))



def test_1d():
    """ Three 1d datafiles """
    # 0, 2, 4, ..., 14, 16 -> resolution=2, window=1
    # 0, 1, 2, ..., 15, 16 -> resolution=1, window=2
    # 0, 3, 6, ..., 12, 15 -> resolution=3, window=0
    expected = [
        [[1], [2],  [1]], [[1], [3],  [1]], [[1], [4],  [1]],
        [[2], [2],  [1]], [[2], [3],  [1]], [[2], [4],  [1]],
        [[3], [5],  [2]], [[3], [6],  [2]], [[3], [7],  [2]],
        [[4], [8],  [3]], [[4], [9],  [3]], [[4], [10], [3]],
        [[5], [8],  [3]], [[5], [9],  [3]], [[5], [10], [3]],
        [[6], [11], [4]], [[6], [12], [4]], [[6], [13], [4]],
        [[7], [14], [5]],
    ]
    check_outputs(configs['1d'], expected)


def test_3d():
    """ Two 3d datafiles """
    # [var_0, var_1], [0, 1, 2], [0, 2, 4],       [0, 2, 4]       -> 2x3x3
    # [var_0],        [0, 1, 2], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4] -> 2x3x3
    expected = [ # time, lat, lon
        [[1, 1, 1],[1, 1, 1]], [[1, 1, 1],[1, 1, 2]], [[1, 1, 1],[1, 1, 3]],
        [[1, 1, 1],[1, 2, 1]], [[1, 1, 1],[1, 2, 2]], [[1, 1, 1],[1, 2, 3]],
        [[1, 1, 1],[1, 3, 1]], [[1, 1, 1],[1, 3, 2]], [[1, 1, 1],[1, 3, 3]],

        [[2, 1, 1],[2, 1, 1]], [[2, 1, 1],[2, 1, 2]], [[2, 1, 1],[2, 1, 3]],
        [[2, 1, 1],[2, 2, 1]], [[2, 1, 1],[2, 2, 2]], [[2, 1, 1],[2, 2, 3]],
        [[2, 1, 1],[2, 3, 1]], [[2, 1, 1],[2, 3, 2]], [[2, 1, 1],[2, 3, 3]],
    ]
    check_outputs(configs['3d'], expected)


def test_3d_nan():
    """ Same 3d data but with a few nan values included """
    def update_data(data):
        data[0].var_0[0,1,2] = np.nan
        data[0].var_1[0,2,0] = np.nan
        data[1].var_0[1,3,1] = np.nan
        return data

    # [var_0, var_1], [0, 1, 2], [0, 2, 4],       [0, 2, 4]       -> 2x3x3
    # [var_0],        [0, 1, 2], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4] -> 2x3x3
    expected = [ # time, lat, lon
        [[2, 1, 1],[2, 1, 1]], [[2, 1, 1],[2, 1, 2]], [[2, 1, 1],[2, 1, 3]],
        [[2, 1, 1],[2, 2, 3]], 
        [[2, 1, 1],[2, 3, 3]],
    ]
    check_outputs(configs['3d'], expected, update_data)


def test_static():
    """ 3d + 2d without temporal dimension (only spatial); asymetric depth """
    na = np.nan
    expected = [ # time, lat, lon
        [[1, 1, 1],[na, 2, 1]], [[1, 1, 1],[na, 2, 2]], [[1, 1, 1],[na, 2, 3]],
        [[1, 1, 1],[na, 3, 1]], [[1, 1, 1],[na, 3, 2]], [[1, 1, 1],[na, 3, 3]], 
        [[2, 1, 1],[na, 2, 1]], [[2, 1, 1],[na, 2, 2]], [[2, 1, 1],[na, 2, 3]],
        [[2, 1, 1],[na, 3, 1]], [[2, 1, 1],[na, 3, 2]], [[2, 1, 1],[na, 3, 3]],
    ]
    check_outputs(configs['static'], expected)


def test_single():
    # 0, 1, 2, ..., 6, 7 -> resolution=1, window=1
    expected = [ [[1]], [[2]], [[3]], [[4]], [[5]], [[6]] ]
    check_outputs(configs['single'], expected)