import pytest
import dask.array as da 
import xarray as xr 
import numpy as np 

from crest.data.loading import Block, Blockset, Datafile, Sample


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
    'data'          : [da.from_array(np.rollaxis(example_data, 0, 3))],
    'coords'        : da.from_array(np.rollaxis(example_coords, 0, 3)),
    'valid_mask'    : da.from_array(np.rollaxis(np.isfinite(example_data), 0, 3)),
    'inbound_mask'  : da.from_array(np.rollaxis(np.ones_like(example_data, dtype=bool), 0, 3)),
    'overlap_mask'  : da.from_array(np.rollaxis(np.zeros_like(example_data, dtype=bool), 0, 3)),
    'coord_vecs'    : [da.from_array([nan, 2., 3.]),
                       da.from_array([0.1, 0.2, 0.3])],
    'dims'          : ['x', 'y'],
    'original_dims' : (['x','y','features'], ['var_1'], {k:float for k in ['x','y','var_1']}),
    'resolution'    : [1., 0.1],
    'window_depth'  : {'x': np.array([0,1]), 'y': np.array([1,0])},
    'valid_percent' : {('x','y'): 1.},
})
example_block4 = Block(**{
    'data'          : [da.from_array(np.rollaxis(example_data2, 0, 3))],
    'coords'        : da.from_array(np.rollaxis(example_coords2, 0, 3)),
    'valid_mask'    : da.from_array(np.rollaxis(np.isfinite(example_data2), 0, 3)),
    'inbound_mask'  : da.from_array(np.rollaxis(np.ones_like(example_data2, dtype=bool), 0, 3)),
    'overlap_mask'  : da.from_array(np.rollaxis(np.zeros_like(example_data2, dtype=bool), 0, 3)),
    'coord_vecs'    : [da.from_array([nan, 2., 3.]),
                       da.from_array([nan, 0.05, 0.15, 0.25, 0.35])],
    'dims'          : ['x', 'y'],
    'original_dims' : (['x','y','features'], ['var_1'], {k:float for k in ['x','y','var_1']}),
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


# =============================================================================
# Automatically generated tests
# =============================================================================


@pytest.mark.parametrize('block', [example_block3, example_block4])
def test_example_blocks_use_prepared_inputs(block: Block) -> None:
    """Keep direct Block fixtures aligned with Datafile's prepared inputs."""
    assert isinstance(block._data, list)
    assert len(block._data) == len(block.features)
    assert all(isinstance(array, da.Array) for array in block._data)
    assert all(array.shape == block.shape + (1,) for array in block._data)
    assert block._valid_mask.shape == block.shape + (1,)
    assert len(block.coord_vecs) == block.ndim
    for vector, size in zip(block.coord_vecs, block.shape):
        assert vector.shape == (size,)

    indices = block.valid_windows
    expected = block._coords.compute()[tuple(indices)]
    np.testing.assert_array_equal(block.valid_coords, expected)


def _prepared_datafile(
    values: np.ndarray, as_sparse: bool, window: tuple[int, int],
) -> Datafile:
    """Create a small source with the same summary metadata as cached data."""
    import sparse

    features = ['v', 'w']
    arrays   = values
    if as_sparse:
        arrays = sparse.COO.from_numpy(values, fill_value=np.nan)
    data = {key: (['x'], da.from_array(arrays[:, i], chunks=-1))
            for i, key in enumerate(features)}
    source = Datafile(xr.Dataset(**{
        'data_vars' : data,
        'coords'    : {'x': np.arange(len(values), dtype=float)},
    }), **{
        'features'     : features,
        'window_depth' : {'x': window},
        'match_radius' : {'x': 0.01},
    })
    source.summary = xr.DataArray(da.zeros((2, 1), chunks=-1), **{
        'dims'   : ['features', 'statistics'],
        'coords' : {'features': features, 'statistics': ['null']},
    })
    source.update_blocks([1])
    return source


@pytest.mark.parametrize('as_sparse', [False, True])
@pytest.mark.parametrize('window', [(0, 0), (1, 0)])
@pytest.mark.parametrize('optimize', [False, True])
def test_prepared_factories_match_multiple_features(
    as_sparse: bool, window: tuple[int, int], optimize: bool,
) -> None:
    """Exercise the typed Datafile -> Blockset -> Sample construction path."""
    from crest.data.loading import SparseBlock

    values  = np.arange(8., dtype=float).reshape(4, 2)
    sources = [_prepared_datafile(values, False, window),
               _prepared_datafile(values + 20., as_sparse, window)]
    factories = [next(iter(source.apply_overlap(0, optimize=optimize)))
                 for source in sources]
    blockset = Blockset(factories)
    assert type(blockset[0]) is Block
    assert type(blockset[1]) is (SparseBlock if as_sparse else Block)
    for block in blockset:
        assert isinstance(block._data, list)
        assert len(block._data) == 2
        assert all(array.shape[-1] == 1 for array in block._data)
        assert block._valid_mask.shape == (4, 1)
        assert len(block.coord_vecs) == 1

    # Matching and raw extraction must retain both source features and coords.
    output = blockset.find_matches(method='tree', seed=0).compute()
    centers = np.arange(window[0], len(values) - window[1])
    assert len(output) == len(centers)
    for sample, center in zip(output, centers):
        assert isinstance(sample, Sample)
        for index, offset in enumerate([0., 20.]):
            expected = values[center - window[0]:center + window[1] + 1]
            for feature, key in enumerate(['v', 'w']):
                np.testing.assert_array_equal(sample[index][key],
                                              expected[:, feature] + offset)
            np.testing.assert_array_equal(sample[index].x,
                np.arange(center - window[0], center + window[1] + 1))
