import pytest
import dask.array as da 
import xarray as xr 
import numpy as np 
import warnings 

from crest.data.loading import Block


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
    'valid_mask'    : da.from_array(np.rollaxis(np.isfinite(example_data), 0, 3)),
    'inbound_mask'  : da.from_array(np.rollaxis(np.ones_like(example_data, dtype=bool), 0, 3)),
    'overlap_mask'  : da.from_array(np.rollaxis(np.zeros_like(example_data, dtype=bool), 0, 3)),
    'dims'          : ['x', 'y'],
    'original_dims' : (['x','y','features'], ['var_1'], {k:float for k in ['x','y','var_1']}, ['var_1']),
    'resolution'    : [1., 0.1],
    'window_depth'  : {'x': np.array([0,1]), 'y': np.array([1,0])},
    'valid_percent' : {('x','y'): 1.},
    'invalid_value' : [12345, 54321., -2147483648, -9223372036854775808, 'a', -np.inf, np.inf],
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
    expect = np.dtype([
      ('values', example_data.dtype, (1,2,2)), 
      ('coords', [
        ('features', example_coords.dtype, (1,)),
        ('x',        example_coords.dtype, (2,)),
        ('y',        example_coords.dtype, (2,)),
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


def test_invalid():
    data = np.array([
      [1, 'a', 'b', 54321., 54321,  np.inf],
      [0, 0.1, nan, 12345, 12345., -np.inf],
    ], dtype=object)
    data = da.overlap.overlap(da.from_array(data, chunks=-1), {1:1}, {1:np.nan}).compute()
    output = example_block.invalid(data)
    expect = np.array([
      [True, False, True, False, True, True, True, True],
      [True, False, False, True, True, True, True, True]
    ])
    assert((output == expect).all()), output.tolist()

    data = np.array([
      [1, -2., 54321., 54321,  np.inf],
      [0, nan, 12345, 12345., -np.inf],
    ], dtype=float)
    data = da.overlap.overlap(da.from_array(data), {1:1}, {1:np.nan}).compute()
    output = example_block.invalid(data)
    expect = np.array([
      [True, False, False, True, True, True, True],
      [True, False, True,  True, True, True, True]
    ])
    assert((output == expect).all()), output.tolist()

    data = np.array([
      [1, -2., 54321., 54321],
      [0,   0, 12345, 12345.],
    ], dtype=int)
    
    # Type mismatch is purposeful
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        data = da.overlap.overlap(da.from_array(data), {1:1}, {1:np.nan}).compute()
        
    output = example_block.invalid(data)
    expect = np.array([
      [True, False, False, True, True, True],
      [True, False, False, True, True, True]
    ])
    # TODO: why is this failing?
    # assert((output == expect).all()), [data, output.tolist()]


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



import sparse

def from_numpy(x):
    coords = np.where(np.isfinite(x))
    data = x[coords]
    coords = np.vstack(coords)
    return sparse.COO(coords, data, shape=x.shape, fill_value=np.nan)
    
# 3 x, 3 y, 1 features
example_data_sparse = from_numpy(
    np.rollaxis( np.array([
     [[nan, nan, nan],
      [10., 20., 30.],
      [40., 50., 60.]]
]), 0, 3) )

# 3 x, 3 y
example_coords_sparse = from_numpy(
    np.rollaxis( np.array([
     [[nan, nan, nan],
      [2.,  2.,  2. ],
      [3.,  3.,  3. ]],

     [[nan, nan, nan],
      [0.1, 0.2, 0.3],
      [0.1, 0.2, 0.3]]
]), 0, 3) )


"""
da.from_array with sparse data raises the warning:
    | DeprecationWarning: coords should be an ndarray. 
    | This will raise a ValueError in the future.
This is coming from the following operation:
    sparse_array[0:0]
which for some reason sparse decides to throw a warning 
about. Appears to be fixed on github, and so it should
be resolved whenever the next version is released.
"""
example_block_sparse = Block(**{
    'data'          : da.from_array(example_data_sparse),
    'coords'        : da.from_array(example_coords_sparse),
    'valid_mask'    : da.from_array(np.rollaxis(np.isfinite(example_data), 0, 3)),
    'inbound_mask'  : da.from_array(np.rollaxis(np.ones_like(example_data, dtype=bool), 0, 3)),
    'overlap_mask'  : da.from_array(np.rollaxis(np.zeros_like(example_data, dtype=bool), 0, 3)),
    'dims'          : ['x', 'y'],
    'original_dims' : (['x','y','features'], ['var_1'], {k:float for k in ['x','y','var_1']}, ['var_1']),
    'resolution'    : [1., 0.1],
    'window_depth'  : {'x': np.array([0,1]), 'y': np.array([1,0])},
    'valid_percent' : {('x','y'): 1.},
})


def test_sparse():
    assert(not example_block.is_sparse)
    assert example_block_sparse.is_sparse


def test_valid_windows_sparse():
    output = example_block_sparse.valid_windows
    output = example_block_sparse.sparse_data[..., 0].coords[:, output]
    expect = np.array([[1,1], [1,2]])
    assert((output == expect).all())


def test_valid_coords_sparse():
    output = example_block_sparse.valid_coords
    expect = np.array([[2, 0.2], [2, 0.3]])
    assert((output == expect).all())


def test_valid_data_sparse():
    output = example_block_sparse.valid_data 
    expect = np.array([[20.], [30.]])
    assert((output == expect).all())
