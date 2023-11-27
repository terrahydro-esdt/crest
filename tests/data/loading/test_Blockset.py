import pytest
import dask.array as da 
import xarray as xr 
import numpy as np 

from crest.data.loading import Block, Blockset, Sample


nan = np.nan 

# 3 x, 3 y, 1 features
example_data = np.array([
     [[nan, nan, nan],
      [10., 20., 30.],
      [40., 50., 60.]]
])
example_data2 = np.array([
     [[nan,  nan,  nan,  nan,  nan],
      [nan,  nan, -20., -30., -40.],
      [nan, -50., -60., -70.,  nan]]
])

# 3 x, 3 y
example_coords = np.array([
     [[nan, nan, nan],
      [2.,  2.,  2. ],
      [3.,  3.,  3. ]],

     [[nan, nan, nan],
      [0.1, 0.2, 0.3],
      [0.1, 0.2, 0.3]]
])
example_coords2 = np.array([
     [[nan, nan, nan, nan, nan],
      [nan,  2.,  2.,  2.,  2.],
      [nan,  3.,  3.,  3.,  3.]],

     [[nan, nan, nan, nan, nan],
      [nan, 0.05, 0.15, 0.25, 0.35],
      [nan, 0.05, 0.15, 0.25, 0.35]]
])

example_block3 = Block(**{
    'data'          : da.from_array(np.rollaxis(example_data, 0, 3)),
    'coords'        : da.from_array(np.rollaxis(example_coords, 0, 3)),
    'inbound_mask'  : da.from_array(np.rollaxis(np.ones_like(example_data, dtype=bool), 0, 3)),
    'overlap_mask'  : da.from_array(np.rollaxis(np.zeros_like(example_data, dtype=bool), 0, 3)),
    'dims'          : ['x', 'y'],
    'original_dims' : (['x','y','features'], ['var_1']),
    'resolution'    : [1., 0.1],
    'window_depth'  : {'x': np.array([0,1]), 'y': np.array([1,0])},
    'valid_percent' : {('x','y'): 1.},
})
example_block4 = Block(**{
    'data'          : da.from_array(np.rollaxis(example_data2, 0, 3)),
    'coords'        : da.from_array(np.rollaxis(example_coords2, 0, 3)),
    'inbound_mask'  : da.from_array(np.rollaxis(np.ones_like(example_data2, dtype=bool), 0, 3)),
    'overlap_mask'  : da.from_array(np.rollaxis(np.zeros_like(example_data2, dtype=bool), 0, 3)),
    'dims'          : ['x', 'y'],
    'original_dims' : (['x','y','features'], ['var_1']),
    'resolution'    : [1., 0.1],
    'window_depth'  : {'x': np.array([0,1]), 'y': np.array([1,0])},
    'valid_percent' : {('x','y'): 1.},
})

example_blockset = Blockset([example_block3, example_block4])


def test_find_matches():
    matches = example_blockset.find_matches()
    assert(matches.shape == (2,))
    assert(matches.dtype['Data_0']['values'].shape == (1,2,2))
    assert(matches.dtype['Data_1']['values'].shape == (1,2,2))

    samples = matches.compute()
    assert(isinstance(samples, np.ndarray))
    assert(all(isinstance(s, Sample) for s in samples))

    exp_1_1  = xr.Dataset(
        data_vars = {'var_1':
            (['x', 'y'], 
            [[10., 20.],
             [40., 50.]]) },
        coords = {
            'x' : [2.,  3.],
            'y' : [0.1, 0.2],
        },
    )
    exp_1_2  = xr.Dataset(
        data_vars = {'var_1':
            (['x', 'y'], 
            [[-20., -30.],
             [-60., -70.]]) },
        coords = {
            'x' : [  2.,   3.],
            'y' : [0.15, 0.25],
        },
    )

    exp_2_1  = xr.Dataset(
        data_vars = {'var_1':
            (['x', 'y'], 
            [[20., 30.],
             [50., 60.]]) },
        coords = {
            'x' : [2.,  3.],
            'y' : [0.2, 0.3],
        },
    )
    exp_2_2  = xr.Dataset(
        data_vars = {'var_1':
            (['x', 'y'], 
            [[-20., -30.],
             [-60., -70.]]) },
        coords = {
            'x' : [2.,  3.],
            'y' : [0.15, 0.25],
        },
    )

    for output, expect in zip(samples, [(exp_1_1, exp_1_2), (exp_2_1, exp_2_2)]):
        for o, e in zip(output, expect):
            assert((o==e).all())
