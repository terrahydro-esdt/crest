from collections.abc import Callable

import dask.array as da
import numpy as np
import pytest

from crest.data.loading import Block, SparseBlock
from .test_Block import _block_kwargs, example_block, example_data, nan



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
example_block_sparse = SparseBlock(**{
    'data'          : [da.from_array(example_data_sparse)],
    'coords'        : da.from_array(example_coords_sparse),
    'valid_mask'    : da.from_array(
        np.isfinite(example_data_sparse).all(-1, keepdims=True)),
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


def test_sparse():
    assert type(example_block) is Block
    assert type(example_block_sparse) is SparseBlock


def test_valid_windows_sparse():
    output = example_block_sparse.valid_windows
    expect = np.array([[1,1], [1,2]])
    assert((output == expect).all())


def test_valid_coords_sparse():
    output = example_block_sparse.valid_coords
    expect = np.array([[2, 0.2], [2, 0.3]])
    assert((output == expect).all())


def test_valid_data_sparse():
    output, _ = example_block_sparse.extract(np.array([0, 1]), ['var_1'])
    output = output[:, :, 0, 1]
    expect = np.array([[20.], [30.]])
    assert((output == expect).all())


# =============================================================================
# Automatically generated tests
# =============================================================================


def _sparse_kwargs(values: np.ndarray, **kwargs) -> dict:
    """Build small sparse fixtures with the same metadata as dense fixtures."""
    result = _block_kwargs(values)
    data = da.from_array(from_numpy(values), chunks=-1)
    result['data'] = [data[..., i:i+1] for i in range(values.shape[-1])]
    coords = result['coords'].compute()
    coords[~np.isfinite(values).any(-1)] = np.nan
    result['coords'] = da.from_array(from_numpy(coords), chunks=-1)
    result['valid_mask'] = da.from_array(
        np.isfinite(from_numpy(values)).all(-1, keepdims=True))
    result |= kwargs
    data = result['data']
    if isinstance(data, da.Array):
        result['data'] = [data[..., i:i+1] for i in range(data.shape[-1])]
    return result


def _sparse_reference(block: SparseBlock, values: np.ndarray) -> np.ndarray:
    """Enumerate sparse candidate/window validity independently on tiny grids."""
    from itertools import product

    candidates = np.isfinite(values).any(-1)
    valid = np.isfinite(values).all(-1)

    if block.is_required:
        for keys, percent in block.valid_percent.items():
            count   = int(np.prod([block.window_total[key] for key in keys]))
            require = int(np.ceil(np.nextafter(percent*count, 0.)))
            updated = np.zeros(block.shape, dtype=bool)
            offsets = [range(-block.window_depth[dim][0],
                             block.window_depth[dim][1] + 1)
                       if dim in keys else (0,) for dim in block.dims]

            for center in np.ndindex(block.shape):
                present = 0
                for offset in product(*offsets):
                    index = tuple(a + b for a, b in zip(center, offset))
                    inside = all(0 <= i < size for i, size in
                                 zip(index, block.shape))
                    present += int(valid[index]) if inside else 0
                updated[center] = present >= require
            valid = updated
    else:
        valid = np.ones(block.shape, dtype=bool)

    # Sparse sampling only exposes source-supported centers.
    valid &= candidates

    # Sparse windows must fit fully inside the logical block.
    for axis, dim in enumerate(block.dims):
        lo, hi = block.window_depth[dim]
        indices = np.arange(block.shape[axis])
        shape = [1] * block.ndim
        shape[axis] = block.shape[axis]
        inside = (indices >= lo) & (indices < block.shape[axis] - hi)
        valid &= inside.reshape(shape)

    return np.array(np.where(valid), dtype=np.intp)


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('groups', [
    {('x', 'y'): 1.},
    {('x', 'y'): 0.4},
    {('x',): 0.5, ('y',): 1.},
    {('y',): 0.5, ('x',): 0.5},
    {('x',): 0.},
])
def test_generated_sparse_window_reference(seed: int, groups: dict) -> None:
    """Exercise grouped rules without quadratic coordinate comparisons."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(7, 6, 2))
    values[rng.random(values.shape[:-1]) < 0.2] = np.nan
    block = SparseBlock(**_sparse_kwargs(values, **{
        'window_depth'  : {'x': (1, 2), 'y': (1, 0)},
        'valid_percent' : groups,
    }))
    expected = _sparse_reference(block, values)
    np.testing.assert_array_equal(block.valid_windows, expected)
    np.testing.assert_array_equal(block.valid_coords, expected.T)


@pytest.mark.parametrize('depth', [(1, 0), [1, 0], np.array([1, 0])])
@pytest.mark.parametrize('sparsity', [0, 0., np.float64(0.)])
def test_sparse_constructor_annotations(
    depth: object, sparsity: object,
) -> None:
    """Inherited constructors retain runtime validation of accepted inputs."""
    block = SparseBlock(**_sparse_kwargs(np.ones((3, 1)), **{
        'window_depth' : {'x': depth},
        'sparsity'     : sparsity,
    }))
    assert type(block) is SparseBlock
    assert isinstance(block, Block)
    assert block.window_total == {'x': 2}


def test_single_origin() -> None:
    """An all-zero coordinate index is a valid sample, not an empty selector."""
    values = np.full((4, 3, 2), np.nan)
    values[0, 0] = [0., 10.]
    kwargs = _sparse_kwargs(values)
    block = SparseBlock(**kwargs)
    np.testing.assert_array_equal(block.valid_windows, [[0], [0]])
    np.testing.assert_array_equal(block.valid_coords, [[0., 0.]])
    output, _ = block.extract(np.array([0]), ['var_0', 'var_1'])
    np.testing.assert_array_equal(output[..., 0, 0], [[0., 10.]])


@pytest.mark.parametrize('use_window', [False, True])
@pytest.mark.parametrize('feature_path', [
    None, ['var_1'], ['y', 'var_0'], ['x'],
])
def test_sparse_extraction_paths(use_window: bool,
                                 feature_path: list | None) -> None:
    """Compare sparse extraction directly with dense source slices."""
    values = np.arange(7*6*2, dtype=float).reshape(7, 6, 2)
    values[0] = np.nan
    depths = {'x': (1, 1), 'y': (0, 1)} if use_window else {}
    kwargs = _sparse_kwargs(values, window_depth=depths)
    block   = SparseBlock(**kwargs)
    centers = block.valid_windows.copy()
    matches = np.array([3, 0, 3, 1])
    result  = block.extract(matches, feature_path)
    for row, match in enumerate(matches):
        x, y = centers[:, match].astype(int)
        left = [x-1, x, x+1] if use_window else [x]
        up   = [y, y+1] if use_window else [y]
        truth = values[np.ix_(left, up)]
        if feature_path is None:
            window = result[row][0]
            np.testing.assert_array_equal(
                window['data'], truth.transpose(2, 0, 1))
            np.testing.assert_array_equal(window['coords']['x'], left)
            np.testing.assert_array_equal(window['coords']['y'], up)
        else:
            array, types = result
            for column, key in enumerate(types):
                if key.startswith('var_'):
                    expected = truth[..., int(key.split('_')[-1])]
                elif key == 'x':
                    expected = np.broadcast_to(
                        np.array(left)[:, None], truth.shape[:2])
                else:
                    expected = np.broadcast_to(up, truth.shape[:2])
                np.testing.assert_array_equal(array[row, column], expected)


def test_sparse_candidates_use_any_feature_support() -> None:
    """Zero-percent sampling keeps centers supported by any source feature."""
    values = np.array([[np.nan, 10.], [1., np.nan], [2., 12.],
                       [np.nan, np.nan]])
    block = SparseBlock(**_sparse_kwargs(values, **{
        'valid_percent' : {('x',): 0.},
    }))
    np.testing.assert_array_equal(block.valid_windows, [[0, 1, 2]])


def test_sparse_uses_prepared_valid_mask() -> None:
    """Feature-validity policy is supplied by the prepared validity mask."""
    values = np.array([[0., np.nan], [np.nan, 1.], [2., 3.]])
    mask = np.isfinite(from_numpy(values)).any(-1, keepdims=True)
    block = SparseBlock(**_sparse_kwargs(values, **{
        'valid_mask' : da.from_array(mask),
    }))
    np.testing.assert_array_equal(block.valid_windows, [[0, 1, 2]])


@pytest.mark.parametrize('feature_path', [None, ['var_0', 'var_1']])
def test_masked_indices_map_to_original_storage(
    feature_path: list | None,
) -> None:
    """Different feature supports must not be indexed by another COO offset."""
    values = np.array([[0., 10.], [1., np.nan], [2., 12.],
                       [np.nan, 13.], [4., 14.]])
    block = SparseBlock(**_sparse_kwargs(values))
    np.testing.assert_array_equal(block.valid_windows, [[0, 2, 4]])
    result = block.extract(np.array([2, 0, 1]), feature_path)
    if feature_path is None:
        output = np.array([np.array(row[0]['data'])[:, 0] for row in result])
    else:
        output = result[0][..., 0]
    np.testing.assert_array_equal(output, values[[4, 0, 2]])


@pytest.mark.parametrize('use_window', [False, True])
def test_sparse_feature_mask_copy_and_reset(use_window: bool) -> None:
    """Mask changes isolate copies without modifying immutable COO arrays."""
    kwargs = _sparse_kwargs(np.arange(7.)[:, None], **{
        'window_depth' : {'x': (1, 0)} if use_window else {},
    })
    block  = SparseBlock(**kwargs)
    before = block.valid_windows.copy()
    copied = block.copy()
    assert type(copied) is SparseBlock
    copied.set_feature_masks({'var_0': (2., 6.)})
    expected = [[3, 4, 5]] if use_window else [[2, 3, 4, 5]]
    np.testing.assert_array_equal(copied.valid_windows, expected)
    np.testing.assert_array_equal(block.valid_windows, before)
    np.testing.assert_array_equal(copied.data.todense(),
                                  np.arange(7.)[:, None])


def test_sparse_percent_invalidation() -> None:
    """Changing percentages must recalculate stored-center selectors."""
    values = np.array([1., 2., np.nan, 4., 5.])[:, None]
    block = SparseBlock(**_sparse_kwargs(values, **{
        'window_depth' : {'x': (1, 1)},
    }))
    original = block.valid_windows.copy()
    assert not original.any()
    block.set_valid_percents({'x': 0.5})
    np.testing.assert_array_equal(block.valid_windows, [[1, 3]])


@pytest.mark.parametrize('use_window', [False, True])
def test_sparse_empty(use_window: bool) -> None:
    """Empty COO storage must produce correctly shaped empty outputs."""
    values = np.full((5, 4, 2), np.nan)
    block = SparseBlock(**_sparse_kwargs(values, **{
        'window_depth' : {'x': (1, 1)} if use_window else {},
    }))
    assert block.valid_coords.shape == (0, 2)
    assert block.fast_invalid_check
    assert block.extract(np.array([], dtype=int)) == []
    direct, _ = block.extract(np.array([], dtype=int), ['var_0'])
    assert direct.shape == ((0, 1, 3, 1) if use_window else (0, 1, 1, 1))


def test_sparse_virtual_dimension() -> None:
    """Raw sparse windows must collapse a virtual axis without axis errors."""
    values = np.arange(5.).reshape(5, 1, 1)
    kwargs = _sparse_kwargs(values)
    _, features, types = kwargs['original_dims']
    kwargs['original_dims'] = (['x', 'features'], features, types)
    block = SparseBlock(**kwargs)
    result = block.extract(np.array([0, 2]))
    assert result[0][0]['data'][0].shape == (1,)
    np.testing.assert_array_equal(result[1][0]['coords']['x'], [2.])


@pytest.mark.parametrize('feature_path', [None, ['var_0', 'x']])
def test_sparse_missing_templates_without_reads(
    feature_path: list | None,
) -> None:
    """Empty optional sparse sources must not touch source data."""
    import dask

    @dask.delayed
    def forbidden() -> sparse.COO:
        raise AssertionError('Unexpected sparse computation')

    kwargs = _sparse_kwargs(np.ones((5, 1)))
    kwargs['data'] = da.from_delayed(forbidden(), **{
        'shape' : (5, 1),
        'meta'  : sparse.COO.from_numpy(np.empty((0, 0)), fill_value=np.nan),
    })
    kwargs['data'] = [kwargs['data']]
    block = SparseBlock(**kwargs)
    result = block.extract(np.array([np.nan]), feature_path)
    assert result is not None
    result = block.extract(np.array([0]), feature_path, empty=True)
    assert result is not None



def test_sparse_logical_grid_stays_sparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a multi-terabyte logical source sparse during bounded extraction."""
    original = sparse.COO.todense

    def bounded(array: sparse.COO) -> np.ndarray:
        assert array.size <= 128, f'Logical array densified: {array.shape}'
        return original(array)

    monkeypatch.setattr(sparse.COO, 'todense', bounded)
    shape = (1_000_000, 1_000_000, 2)
    index = np.array([[0, 0, 999999, 999999],
                      [0, 0, 999998, 999998], [0, 1, 0, 1]])
    source = sparse.COO(index, np.array([0., 10., 20., 30.]), **{
        'shape'      : shape,
        'fill_value' : np.nan,
    })
    block = SparseBlock(**{
        'data'          : [da.from_array(source[..., i:i+1], chunks=-1)
                           for i in range(2)],
        'coords'        : da.empty(shape, chunks=(1000, 1000, 2)),
        'inbound_mask'  : da.ones(shape[:-1] + (1,), chunks=(1000, 1000, 1)),
        'overlap_mask'  : da.zeros(shape[:-1] + (1,), chunks=(1000, 1000, 1)),
        'valid_mask'    : da.from_array(
            np.isfinite(source).all(-1, keepdims=True)),
        'dims'          : ['x', 'y'],
        'original_dims' : (['x', 'y', 'features'], ['a', 'b'],
                           dict.fromkeys(['x', 'y', 'a', 'b'], float),
                        ),
        'resolution'    : [1., 1.],
        'valid_percent' : {('x', 'y'): 1.},
        'coord_vecs'    : [da.arange(1_000_000., chunks=1000)] * 2,
    })
    np.testing.assert_array_equal(block.valid_coords,
                                  [[0., 0.], [999999., 999998.]])
    values, _ = block.extract(np.array([1, 0]), ['a', 'b'])
    np.testing.assert_array_equal(values[..., 0, 0], [[20., 30.], [0., 10.]])
    assert isinstance(block.data, sparse.COO)


def test_sparse_cleanup() -> None:
    """Cleanup discards derived caches while retaining sparse payload data."""
    block = SparseBlock(**_sparse_kwargs(np.arange(5.)[:, None], **{
        'window_depth' : {'x': (1, 0)},
    }))
    _ = block.valid_coords, block.coords
    payload = block.data
    block.cleanup()
    assert not {'valid_mask', 'valid_windows', 'coords',
                'fast_invalid_check'} & block.__dict__.keys()
    assert block.data is payload


@pytest.mark.speed
def test_sparse_extraction_benchmark(record_property: Callable) -> None:
    """Record point-extraction throughput without absolute timing limits."""
    from time import perf_counter

    values = np.full((128, 128, 4), np.nan)
    values[::4, ::4] = 1.
    block = SparseBlock(**_sparse_kwargs(values))
    matches = np.arange(256)
    block.extract(matches, ['var_0'])
    start = perf_counter()
    for _ in range(10): block.extract(matches, ['var_0'])
    record_property('seconds_per_extract', (perf_counter() - start) / 10)


@pytest.mark.parametrize('optimize', [False, True])
def test_datafile_constructs_sparse_block(optimize: bool) -> None:
    """Sparse final chunk metadata selects SparseBlock at lazy construction."""
    import xarray as xr
    from crest.data.loading import Datafile

    values = np.arange(20., dtype=float).reshape(5, 4)
    values[0] = np.nan
    data = da.from_array(from_numpy(values))
    source = Datafile(xr.Dataset(**{
        'data_vars' : {'v': (['x', 'y'], data)},
        'coords'    : {'x': np.arange(5.), 'y': np.arange(4.)},
    }), **{
        'features'     : ['v'],
        'window_depth' : {'x': 0, 'y': 0},
        'match_radius' : {'x': 0.5, 'y': 0.5},
    })
    source.summary = xr.DataArray(da.from_array([[4.]]), **{
        'dims'   : ['features', 'statistics'],
        'coords' : {'features': ['v'], 'statistics': ['null']},
    })
    source.update_blocks([1, 1])
    deferred = list(source.apply_overlap(0, optimize=optimize))
    block    = deferred[0]()
    assert type(block) is SparseBlock
    assert isinstance(block, Block)
    output, _ = block.extract(np.array([0, 1]), ['v'])
    np.testing.assert_array_equal(output[:, 0, 0, 0], [4., 5.])



@pytest.mark.parametrize('use_window', [False, True])
@pytest.mark.parametrize('scheduler', ['synchronous', 'threads'])
def test_sparse_payload_materialized_once(
    use_window: bool,
    scheduler: str,
) -> None:
    """All sparse read paths share one materialized COO payload."""
    import dask

    calls = []
    values = np.arange(24., dtype=float).reshape(12, 2)
    payload = from_numpy(values)

    @dask.delayed(pure=False)
    def read() -> sparse.COO:
        calls.append(True)
        return payload

    source = da.from_delayed(read(), **{
        'shape' : payload.shape,
        'dtype' : payload.dtype,
        'meta'  : payload,
    })
    block = SparseBlock(**_sparse_kwargs(values, **{
        'data'         : source,
        'window_depth' : {'x': (1, 1) if use_window else (0, 0)},
    }))
    with dask.config.set(scheduler=scheduler):
        _ = block.valid_windows, block.valid_coords, block.coords
        cached = block.data
        for features in [None, ['var_0', 'var_1'], None, ['var_1']]:
            block.extract(np.array([0, 1, 0]), features)
            assert block.data is cached
        assert len(calls) == 1
        assert isinstance(block.data, sparse.COO)
        copied = block.copy()
        assert copied.data is cached
        assert 'valid_coords' not in SparseBlock.__dict__
        for key in ['_unmasked_data', 'unmasked_data', 'sparse_data',
                    'unmasked_sparse_data']:
            assert not hasattr(block, key)

        # Cleanup discards derived arrays but retains the materialized payload.
        block.cleanup()
        assert block.data is cached
        assert copied.data is cached
        assert len(calls) == 1
        _ = block.valid_windows
        assert block.data is cached
        assert len(calls) == 1
        copied.cleanup()


@pytest.mark.parametrize('use_window', [False, True])
def test_sparse_payload_retained_after_cleanup(use_window: bool) -> None:
    """Cleanup preserves the computed COO payload for subsequent reuse."""
    block = SparseBlock(**_sparse_kwargs(np.arange(10.)[:, None], **{
        'window_depth' : {'x': (1, 1) if use_window else (0, 0)},
    }))
    _ = block.valid_windows, block.coords
    payload = block.data
    copied  = block.copy()
    block.cleanup()
    copied.cleanup()
    assert block.data is payload
    assert copied.data is payload


@pytest.mark.parametrize('lazy_vectors', [False, True])
@pytest.mark.parametrize('empty', [False, True])
def test_sparse_matching_uses_shared_vector_lookup(
    lazy_vectors: bool,
    empty: bool,
) -> None:
    """Coordinate-only matching neither gathers payloads nor reads the grid."""
    import dask

    @dask.delayed(pure=False)
    def forbidden() -> np.ndarray:
        raise AssertionError('Shared coordinate lookup triggered a source read')

    values = np.ones((5, 4, 1), dtype='float32')
    vectors = [np.arange(5, dtype='float32') * 3,
               np.array([np.nan, 2., 8., 12.], dtype='float32')]
    kwargs = _sparse_kwargs(values)
    kwargs['data'] = [da.from_delayed(forbidden(), **{
        'shape' : values.shape,
        'dtype' : values.dtype,
    })]
    kwargs['coords'] = da.from_delayed(forbidden(), **{
        'shape' : (5, 4, 2),
        'dtype' : values.dtype,
    })
    kwargs['coord_vecs'] = [da.from_array(v, chunks=2) for v in vectors] \
                          if lazy_vectors else vectors
    block = SparseBlock(**kwargs)
    indices = np.array([[4, 1, 4], [2, 0, 2]])
    if empty: indices = indices[:, :0]
    block.valid_windows = indices
    expected = np.stack([v[i] for v, i in zip(vectors, indices)], axis=-1)
    np.testing.assert_array_equal(block.valid_coords, expected)
    assert block.valid_coords.dtype == np.dtype('float32')
    assert 'coords' not in block.__dict__
    assert 'data' not in block.__dict__


@pytest.mark.parametrize('percent', [0., 0.3, 1.])
def test_sparse_bin_masks_keep_source_values(percent: float) -> None:
    """Combining bin masks does not overwrite stored neighbor values."""
    values = np.array([1., 1., 1., 10., 1., 1., 1.])[:, None]
    block = SparseBlock(**_sparse_kwargs(values, **{
        'window_depth'  : {'x': (1, 1)},
        'valid_percent' : {('x',): percent},
    }))
    payload = block.data
    block.set_feature_masks({'var_0': (0., 2.)})
    centers = [1, 5] if percent == 1 else [1, 2, 4, 5]
    np.testing.assert_array_equal(block.valid_windows, [centers])
    np.testing.assert_array_equal(block.data.todense(), values)
    assert block.data is payload
    if percent < 1:
        output, _ = block.extract(np.array([1]), ['var_0'])
        np.testing.assert_array_equal(output[0, 0], [1., 1., 10.])
