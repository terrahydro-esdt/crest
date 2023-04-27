import pytest
import xarray as xr 
import numpy as np 
import dask.array as da 

from crest.src.data.loading import Block


nan = np.nan 

# 3 x, 3 y, 1 features
example_data = np.array([
     [[nan, nan, nan],
      [10., 20., 30.],
      [40., 50., 60.]]
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

example_block = Block(**{
    'data'          : da.from_array(np.rollaxis(example_data, 0, 3)),
    'coords'        : da.from_array(np.rollaxis(example_coords, 0, 3)),
    'dims'          : ['x', 'y'],
    'original_dims' : (['x','y','features'], ['var_1']),
    'resolution'    : [1., 0.1],
    'window_depth'  : {'x': np.array([0,1]), 'y': np.array([1,0])},
    'valid_percent' : {('x','y'): 1.},
})


def test_data():
    output = np.nan_to_num(example_block.data)
    expect = np.nan_to_num(np.rollaxis(example_data, 0, 3))
    assert((output == expect).all())


def test_coords():
    output = np.nan_to_num(example_block.coords)
    expect = np.nan_to_num(np.rollaxis(example_coords, 0, 3))
    assert((output == expect).all())


def test_dtype():
    output = example_block.dtype 
    expect = np.dtype([('values', np.float32, (1,2,2)), ('coords', [
        ('features', np.float32, (1,)),
        ('x', np.float32, (2,)),
        ('y', np.float32, (2,)),
    ])])
    assert(output == expect)


def test_axes():
    assert(example_block.axes == {'x': 0, 'y': 1})


def test_window_total():
    assert(example_block.window_total == {'x':2, 'y':2})


def test_valid_windows():
    output = example_block.valid_windows
    expect = np.array([[1,1], [1,2]])
    assert((output == expect).all())


def test_valid_coords():
    output = example_block.valid_coords
    expect = np.array([[2, 0.2], [2, 0.3]])
    assert((output == expect).all())


def test_valid_data():
    output = example_block.valid_data 
    expect = np.array([[20.], [30.]])
    assert((output == expect).all())


def test_extract():
    output = example_block.extract(np.arange(len(example_block.valid_windows)))
    exp_1  = xr.Dataset(
        data_vars = {'var_1':
            (['x', 'y'], 
            [[10., 20.],
             [40., 50.]]) },
        coords = {
            'x' : [2.,  3.],
            'y' : [0.1, 0.2],
        },
    )
    exp_2  = xr.Dataset(
        data_vars = {'var_1':
            (['x', 'y'], 
            [[20., 30.],
             [50., 60.]]) },
        coords = {
            'x' : [2.,  3.],
            'y' : [0.2, 0.3],
        },
    )
    for out, exp in zip(output, [exp_1, exp_2]):
        assert((out == exp).all())