from collections import defaultdict as dd
from functools import partial
from itertools import zip_longest

import dask.array as da
import xarray as xr
import numpy as np
import pytest
import dask # Added for patch with forking error

from crest.data.loading import Dataset, Datafile
from crest.utils import synthetic_data
from .Dataset_config import configs


def check_outputs(config, expected, update_data=lambda x: x, norm=True):

    """ Run the given config, and check if outputs match expectations """
    config = config.copy()
    depths = config.pop('depth', [])

    data = update_data(synthetic_data(**config) if config else None)
    dfs  = [Datafile(d, window_depth=wd) for d,wd in zip_longest(data, depths, fillvalue={})]

    with dask.config.set(scheduler='synchronous'): # Added for patch with forking error
        dataset = Dataset(dfs)
        outputs = dataset.generate_samples()

        try:
            for output in outputs.compute():
                print([list(o.x.to_numpy()) for o in output])
        except: pass

        for output, expect in zip(outputs.compute(), expected):
            for out, exp, depth in zip_longest(output, map(tuple, expect), depths, fillvalue={}):
                resolut = getattr(out, 'resolution', dd(lambda: 1)) if norm else dd(lambda: 1)
                get_mid = lambda v, d: (v[d[0]] if isinstance(d, tuple) else v[len(v) // 2]).values.item()
                get_out = lambda d: round(get_mid(out[d], depth.get(d, 0)) / resolut[d], 6)
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
        [[1], [2],  [1]], [[1], [3],  [1]],
        [[2], [3],  [1]], [[2], [4],  [1]],
        [[3], [5],  [2]], [[3], [6],  [2]], [[3], [7],  [2]],
        [[4], [8],  [3]], [[4], [9],  [3]],
        [[5], [9],  [3]], [[5], [10], [3]],
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
    expected = [ # time, lat, lon
        [[1, 1, 1],[2, 1]], [[1, 1, 1],[2, 2]], [[1, 1, 1],[2, 3]],
        [[1, 1, 1],[3, 1]], [[1, 1, 1],[3, 2]], [[1, 1, 1],[3, 3]],
        [[2, 1, 1],[2, 1]], [[2, 1, 1],[2, 2]], [[2, 1, 1],[2, 3]],
        [[2, 1, 1],[3, 1]], [[2, 1, 1],[3, 2]], [[2, 1, 1],[3, 3]],
    ]
    check_outputs(configs['static'], expected)


def test_single():
    # 0, 1, 2, ..., 6, 7 -> resolution=1, window=1
    expected = [ [[1]], [[2]], [[3]], [[4]], [[5]], [[6]] ]
    check_outputs(configs['single'], expected)


def test_two_blocks():
    """ Data chunked into two blocks """
    config = {
        'data_kwargs' : [{'dimensions': {'x': 5}, 'chunks': {'x': 4}}],
        'depth' : [{'x': (3,0)}],
    }
    # 0, 1, 2, 3, 4 -> [(0, 1, 2, 3), (1, 2, 3, 4)]
    expected = [ [[3]], [[4]] ]
    check_outputs(config, expected)

    config = {
        'data_kwargs' : [{'dimensions': {'x': 5}, 'chunks': {'x': 4}}],
        'depth' : [{'x': (0,3)}],
    }
    # 0, 1, 2, 3, 4 -> [(0, 1, 2, 3), (1, 2, 3, 4)]
    expected = [ [[0]], [[1]] ]
    check_outputs(config, expected)

    config = {
        'data_kwargs' : [{'dimensions': {'x': 5}, 'chunks': {'x': 4}}],
        'depth' : [{'x': (1,2)}],
    }
    # 0, 1, 2, 3, 4 -> [(0, 1, 2, 3), (1, 2, 3, 4)]
    expected = [ [[1]], [[2]] ]
    check_outputs(config, expected)

    config = {
        'data_kwargs' : [{'dimensions': {'x': 5}, 'chunks': {'x': 4}}],
        'depth' : [{'x': (2,1)}],
    }
    # 0, 1, 2, 3, 4 -> [(0, 1, 2, 3), (1, 2, 3, 4)]
    expected = [ [[2]], [[3]] ]
    check_outputs(config, expected)


def test_non_uniform():
    """ Test non-uniform grid matching """
    def update_data(_):
        gen_xr = lambda v, c: xr.DataArray(v, coords=c).to_dataset('features')
        return [gen_xr(np.array([v]).T, {'x':v, 'features':[f'var{i}']})
                for i, v in enumerate([
                    [1,          9,     13, 15, 16],
                    [1, 3, 5, 7, 9, 11, 13, 15],
                ])]

    config   = {}
    expected = [
        [[1],[1]],   [[1],[3]],  [[1],[5]],
        [[9],[5]],   [[9],[7]],  [[9],[9]], [[9],[11]],
        [[13],[11]], [[13],[13]],
        [[15],[15]], [[16],[15]],
    ]
    check_outputs(config, expected, update_data, norm=False)



class TestDataset:

    @pytest.fixture(scope='function')
    def synthetic(self):
        # Generate synthetic data
        dims = [ # shape, chunks
            [(100, 200, 30), (85,  50, 1)],
            [( 20,  50,  5), (10,  25, 1)],
            [(     800,  5), (     40, 1)],
            [(     200,  3), (    200, 1)],
            [( 14,  40,  1), ( 8,  25, 1)],
        ]
        ext  = [ # extent (min/max)
            [(1, 25), (0, 60)],
            [(2, 24), (2, 60)],
            [         (1, 61)],
            [         (0, 59)],
            [(0, 25), (0, 60)],
        ]
        data = [da.zeros(shp, chunks=chk, dtype='float32') for shp,chk in dims]
        keys = ['a', 'b']
        feat = lambda n, p: [f'var{p}_{i}' for i in range(n)]

        datafiles = [
            Datafile(xr.DataArray(d, coords={k: np.linspace(*(e+(s,)))
                for e, k, s in zip(ex, keys[-len(ex):], shape)
            } | {'features':feat(shape[-1], i)}).to_dataset('features'),
        ) for i, (d, ex, (shape, _)) in enumerate(zip(data, ext, dims))]
        return Dataset(datafiles)


    @pytest.fixture(autouse=True)
    def ensure_dims(self, synthetic):
        """ Add virtual dimensions """
        synthetic.ensure_dims( set.union(*map(set, synthetic.dims)) )


    def test_ensure_dims(self, synthetic):
        """ Test adding virtual dimensions """
        assert((np.array(synthetic.virtual) == np.array([
            [False, False],
            [False, False],
            [ True, False],
            [ True, False],
            [False, False],
        ])).all()), synthetic.virtual


    def test_autochunk_min(self, synthetic):
        """ Test automatically chunking data with small blocksize """
        synthetic.autochunk(numblocks=560, verbose=True)
        assert((np.array(synthetic.numblocks) == np.array([
            [14, 40, 30],
            [14, 40,  5],
            [ 1, 40,  5],
            [ 1, 40,  3],
            [14, 40,  1],
        ])).all()), synthetic.numblocks


    def test_autochunk_mid(self, synthetic):
        """ Test automatically chunking data with mid blocksize """
        synthetic.autochunk(numblocks=40, verbose=True)
        assert((np.array(synthetic.numblocks) == np.array([
            [2, 20, 30],
            [2, 20,  5],
            [1, 20,  5],
            [1, 20,  3],
            [2, 20,  1],
        ])).all()), synthetic.numblocks


    def test_autochunk_max(self, synthetic):
        """ Test automatically chunking data with large blocksize """
        synthetic.autochunk(numblocks=4, verbose=True)
        assert((np.array(synthetic.numblocks) == np.array([
            [1, 4, 30],
            [1, 4,  5],
            [1, 4,  5],
            [1, 4,  3],
            [1, 4,  1],
        ])).all()), synthetic.numblocks


    def test_create_blocks(self, synthetic):
        """ Test creating the Block objects """
        blocks = synthetic.create_blocks(numblocks=4, verbose=True)
        assert(len(blocks) == 4), len(blocks)



""" Unclear how to prevent duplicate matchups in situations like this test,
    without missing some matchups in other situations (like test_1d).

    It's potentially something along the lines of masking overlapped elements
    conditional upon the window depth (e.g. depth of (0,2) would mask the
    left and right sides differently) - but uncertain at the moment.
"""
# def test_two_blocks_double():
#     """ Data chunked into two blocks for two datasets """
#     config = {
#         'data_kwargs' : [
#             {'dimensions': {'x': 6}, 'chunks': {'x': 2}},
#             {'dimensions': {'x': 3}, 'chunks': {'x': 1}},
#         ],
#         'depth' : [{'x': (2,0)}, {'x': (1,0)}],
#     }
#     # 0, 1, 2, 3, 4 -> [(0, 1, 2, 3), (1, 2, 3, 4)]
#     expected = [ [[2],[1]], [[3],[1]], [[3],[2]], [[4],[2]], [[5],[2]] ]
#     check_outputs(config, expected)
