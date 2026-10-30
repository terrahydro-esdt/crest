from collections.abc import Callable

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
    output, _ = example_block.extract(np.array([0, 1]), ['var_1'])
    expect = np.array([[20.], [30.]])
    assert((output[:, :, 0, 1] == expect).all())


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


# =============================================================================
# Automatically generated tests
# =============================================================================


def _block_kwargs(values: np.ndarray, **kwargs) -> dict:
    """Build fresh, small block inputs without sharing mutable test state."""
    shape    = values.shape[:-1]
    dims     = list('xyz')[:len(shape)]
    features = [f'var_{i}' for i in range(values.shape[-1])]
    coords   = np.moveaxis(np.indices(shape, dtype=float), 0, -1)
    vectors  = [da.arange(size, dtype=float) for size in shape]
    types    = dict.fromkeys(dims + features, values.dtype)
    types.update(dict.fromkeys(dims, np.dtype(float)))
    result = {
        'data'          : da.from_array(values, chunks=-1),
        'coords'        : da.from_array(coords, chunks=-1),
        'inbound_mask'  : da.ones(shape + (1,), dtype=bool, chunks=-1),
        'overlap_mask'  : da.zeros(shape + (1,), dtype=bool, chunks=-1),
        'valid_mask'    : da.from_array(
            np.isfinite(values).all(-1, keepdims=True), chunks=-1),
        'dims'          : dims,
        'original_dims' : (dims + ['features'], features, types),
        'resolution'    : [1.] * len(dims),
        'window_depth'  : dict.fromkeys(dims, (0, 0)),
        'valid_percent' : {tuple(dims): 1.},
        'coord_vecs'    : vectors,
    } | kwargs
    data = result['data']
    if isinstance(data, da.Array):
        result['data'] = [data[..., i:i+1] for i in range(data.shape[-1])]
    return result


def _dense_reference(block: Block, mask: np.ndarray) -> np.ndarray:
    """Enumerate tiny windows independently of the rolling implementation."""
    from itertools import product

    invalid = (~mask[..., 0].copy() if block.is_required else
               np.zeros(block.shape, dtype=bool))

    if block.is_required:
        for keys, percent in block.valid_percent.items():
            sizes   = block.window_total
            count   = int(np.prod([sizes[key] for key in keys]))
            require = int(np.ceil(np.nextafter(percent*count, 0.)))
            maximum = count - min(count, require)
            updated = np.zeros_like(invalid)
            offsets = [range(-block.window_depth[dim][0],
                             block.window_depth[dim][1] + 1)
                       if dim in keys else (0,) for dim in block.dims]
            for center in np.ndindex(block.shape):
                bad = 0
                for offset in product(*offsets):
                    index = tuple(a + b for a, b in zip(center, offset))
                    inside = all(0 <= i < size for i, size in
                                 zip(index, block.shape))
                    bad += int(invalid[index]) if inside else 1
                updated[center] = bad > maximum
            invalid = updated

    # Boundary policy is intentionally separate from value validity.
    result = []
    inbound = block._inbound.compute()[..., 0]
    overlap = block._overlap.compute()[..., 0]
    for center in np.ndindex(block.shape):
        if invalid[center] or not inbound[center]: continue
        if not block.allow_repeats and block.block_index == 0:
            if overlap[center]: continue
        valid = True
        for keys in block.valid_percent:
            for dim in keys:
                axis  = block.dims.index(dim)
                lo,hi = block.window_depth[dim]
                valid &= lo <= center[axis] < block.shape[axis] - hi
        if valid: result.append(center)
    return np.array(result, dtype=np.intp).reshape(-1, block.ndim).T


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('groups', [
    {('x', 'y'): 1.},
    {('x', 'y'): 0.4},
    {('x',): 0.5, ('y',): 1.},
    {('y',): 0.5, ('x',): 0.5},
    {('x',): 0., ('y',): 1.},
])
def test_generated_window_reference(seed: int, groups: dict) -> None:
    """Match exhaustive enumeration for ordered and combined validity rules."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(7, 6, 2))
    values[rng.random(values.shape) < 0.2] = np.nan
    block = Block(**_block_kwargs(values, **{
        'window_depth'  : {'x': (1, 2), 'y': (1, 0)},
        'valid_percent' : groups,
    }))
    expected = _dense_reference(block, block._valid_mask.compute())
    np.testing.assert_array_equal(block.valid_windows, expected)


@pytest.mark.parametrize('depth', [(1, 0), [1, 0], np.array([1, 0])])
@pytest.mark.parametrize('sparsity', [0, 0., np.float64(0.)])
def test_constructor_annotations(depth: object, sparsity: object) -> None:
    """Accept existing producer representations through runtime type checks."""
    block = Block(**_block_kwargs(np.ones((3, 1)), **{
        'window_depth' : {'x': depth},
        'sparsity'     : sparsity,
    }))
    assert block.window_total == {'x': 2}


@pytest.mark.parametrize('length', [65536, 65537, 65633, 65634])
def test_index_width(length: int) -> None:
    """Indices must not wrap at the uint16 boundary."""
    block = Block(**_block_kwargs(np.ones((length, 1))))
    indices = block.valid_windows
    assert indices.shape == (1, length)
    assert int(indices[0, -1]) == length - 1
    assert np.unique(indices).size == length


@pytest.mark.parametrize('percent', [0., 1e-8, 0.01, 0.28, 1.])
def test_fractional_threshold(percent: float) -> None:
    """Preserve fractional thresholds without rounding tiny values to zero."""
    values = np.array([1., np.nan, np.nan, np.nan, 2.])[:, None]
    block = Block(**_block_kwargs(values, **{
        'window_depth'  : {'x': (1, 1)},
        'valid_percent' : {('x',): percent},
    }))
    np.testing.assert_array_equal(block.valid_windows,
        _dense_reference(block, np.isfinite(values)))


def test_fractional_threshold_integral_boundary() -> None:
    """Mathematically integral thresholds must not round one cell too high."""
    values = np.array([1., 1., 1., 1., np.nan, 1., 1.])[:, None]
    block = Block(**_block_kwargs(values, **{
        'window_depth'  : {'x': (2, 2)},
        'valid_percent' : {('x',): 0.8},
    }))

    # Center 2 contains exactly four valid cells out of five.
    assert 2 in block.valid_windows[0]
    np.testing.assert_array_equal(block.valid_windows,
        _dense_reference(block, np.isfinite(values)))


@pytest.mark.parametrize('source_index', [0, 1])
@pytest.mark.parametrize('allow_repeats', [False, True])
def test_overlap_center_convention(source_index: int,
                                  allow_repeats: bool) -> None:
    """Preserve first-source overlap exclusion and inbound clipping."""
    values  = np.ones((5, 1))
    inbound = np.array([1, 1, 1, 1, 0], dtype=bool)[:, None]
    overlap = np.array([1, 0, 0, 0, 0], dtype=bool)[:, None]
    block = Block(**_block_kwargs(values, **{
        'inbound_mask'  : da.from_array(inbound),
        'overlap_mask'  : da.from_array(overlap),
        'block_index'   : source_index,
        'allow_repeats' : allow_repeats,
    }))
    expected = [0, 1, 2, 3] if allow_repeats or source_index else [1, 2, 3]
    np.testing.assert_array_equal(block.valid_windows, [expected])


@pytest.mark.parametrize('percent', [0., 1.])
def test_window_bounds_without_rule(percent: float) -> None:
    """Do not add new clipping along dimensions absent from the policy."""
    block = Block(**_block_kwargs(np.ones((4, 4, 1)), **{
        'window_depth'  : {'x': (1, 0), 'y': (1, 0)},
        'valid_percent' : {('x',): percent},
    }))
    assert (block.valid_windows[0] >= 1).all()
    assert (block.valid_windows[1] == 0).any()


@pytest.mark.parametrize('feature_path', [
    None, ['var_1'], ['y', 'var_0'], ['x'],
])
def test_extraction_paths(feature_path: list | None) -> None:
    """Compare raw and direct outputs against original source slices."""
    values = np.arange(7*6*2, dtype=float).reshape(7, 6, 2)
    kwargs = _block_kwargs(values, **{
        'window_depth' : {'x': (1, 1), 'y': (0, 1)},
    })
    block   = Block(**kwargs)
    centers = block.valid_windows.copy()
    matches = np.array([3, 0, 3, 1])
    result  = block.extract(matches, feature_path)
    for row, match in enumerate(matches):
        x, y = centers[:, match].astype(int)
        truth = values[x-1:x+2, y:y+2]
        if feature_path is None:
            window = result[row][0]
            np.testing.assert_array_equal(
                window['data'], truth.transpose(2, 0, 1))
            np.testing.assert_array_equal(window['coords']['x'], [x-1, x, x+1])
            np.testing.assert_array_equal(window['coords']['y'], [y, y+1])
        else:
            array, types = result
            for column, key in enumerate(types):
                if key.startswith('var_'):
                    expected = truth[..., int(key.split('_')[-1])]
                elif key == 'x':
                    vector   = np.arange(x-1, x+2)[:, None]
                    expected = np.broadcast_to(vector, (3, 2))
                else:
                    expected = np.broadcast_to(np.arange(y, y+2), (3, 2))
                np.testing.assert_array_equal(array[row, column], expected)
    if feature_path is None:
        assert result[0][0] is result[2][0]


@pytest.mark.parametrize('feature_path', [None, ['var_0', 'x']])
def test_original_dimension_order(feature_path: list | None) -> None:
    """Restore declared source order after storage-axis ordering."""
    values = np.arange(5*4, dtype=float).reshape(5, 4, 1)
    kwargs = _block_kwargs(values, **{
        'window_depth' : {'x': (1, 0), 'y': (0, 1)},
    })
    _, features, types = kwargs['original_dims']
    kwargs['original_dims'] = (
        ['y', 'features', 'x'], features, types)
    block   = Block(**kwargs)
    centers = block.valid_windows[:, :1].copy()
    result  = block.extract(np.array([0]), feature_path)
    x, y    = centers[:, 0].astype(int)
    expected = values[x-1:x+1, y:y+2, 0].T
    output = (result[0][0]['data'][0] if feature_path is None else
              result[0][0, 0])
    np.testing.assert_array_equal(output, expected)
    if feature_path is None:
        assert result[0][0]['dims'] == ['features', 'y', 'x']


@pytest.mark.parametrize('feature_path', [None, ['var_0', 'x']])
def test_virtual_dimension(feature_path: list | None) -> None:
    """Omit virtual axes without changing real-axis data or coordinates."""
    values = np.arange(5., dtype=float).reshape(5, 1, 1)
    kwargs = _block_kwargs(values)
    _, features, types = kwargs['original_dims']
    kwargs['original_dims'] = (['x', 'features'], features, types)
    block  = Block(**kwargs)
    result = block.extract(np.array([0, 2]), feature_path)
    if feature_path is None:
        assert result[0][0]['data'][0].shape == (1,)
        assert list(result[0][0]['coords']) == ['x', 'features']
        np.testing.assert_array_equal(result[1][0]['data'][0], [2.])
    else:
        assert result[0].shape == (2, 2, 1)
        np.testing.assert_array_equal(result[0][1, :, 0], [2., 2.])


@pytest.mark.parametrize('feature_path', [None, ['var_0']])
def test_missing_matches_do_not_mutate_input(feature_path: list | None) -> None:
    """Missing entries and repeated groups must not alter caller arrays."""
    block   = Block(**_block_kwargs(np.arange(5.)[:, None]))
    matches = np.array([[2., np.nan], [0., 2.]])
    before  = matches.copy()
    result  = block.extract(matches, feature_path)
    np.testing.assert_array_equal(matches, before)
    if feature_path is None:
        assert np.isnan(result[0][1]['data'][0]).all()
        np.testing.assert_array_equal(result[1][1]['data'][0], [2.])
    else:
        assert result[0].shape == (2, 2, 1, 1)
        assert np.isnan(result[0][0, 1]).all()
        np.testing.assert_array_equal(result[0][1, 1, 0], [2.])


@pytest.mark.parametrize('feature_path', [None, ['var_0']])
@pytest.mark.parametrize('empty', [False, True])
def test_missing_templates_without_reads(feature_path: list | None,
                                        empty: bool) -> None:
    """Missing templates require metadata, not source or validity reads."""
    import dask

    @dask.delayed
    def forbidden() -> np.ndarray:
        raise AssertionError('Unexpected source computation')

    kwargs = _block_kwargs(np.ones((5, 1)))
    for key in ['data', 'coords', 'valid_mask', 'inbound_mask', 'overlap_mask']:
        kwargs[key] = da.from_delayed(forbidden(), shape=(5, 1), dtype=float)
    kwargs['data'] = [kwargs['data']]
    block = Block(**kwargs)
    result = block.extract(np.full(2, np.nan), feature_path, empty=empty)
    if feature_path is None:
        assert len(result) == (1 if empty else 2)
    else:
        assert result[0].shape == ((1, 1) if empty else (2, 1, 1))
        assert np.isnan(result[0]).all()


def test_direct_path_reads_only_requested_values() -> None:
    """A known validity mask must allow skipping unused data and coordinates."""
    import dask

    @dask.delayed
    def forbidden() -> np.ndarray:
        raise AssertionError('Unrequested data were computed')

    kwargs = _block_kwargs(np.ones((5, 2)))
    kwargs['data'] = [da.arange(5.)[:, None],
                     da.from_delayed(forbidden(), shape=(5, 1), dtype=float)]
    kwargs['coords'] = da.from_delayed(forbidden(), shape=(5, 1), dtype=float)
    block = Block(**kwargs)
    data, types = block.extract(np.array([0, 2]), ['var_0'])
    np.testing.assert_array_equal(data[:, 0, 0], [0., 2.])
    assert list(types) == ['var_0']


def test_validity_policy_invalidation() -> None:
    """Changing percentages must discard stale derived masks."""
    block = Block(**_block_kwargs(np.array([1., np.nan, 2.])[:, None], **{
        'window_depth' : {'x': (1, 1)},
    }))
    original = block.valid_windows.copy()
    assert original.shape == (1, 0)
    block.set_valid_percents({'x': 0.})
    np.testing.assert_array_equal(block.valid_windows, [[1]])


def test_feature_masks_and_copy() -> None:
    """Masking a copy must preserve original data and original validity."""
    kwargs = _block_kwargs(np.arange(5.)[:, None])
    block = Block(**kwargs)
    before = block.valid_windows.copy()
    copied = block.copy()
    assert type(copied) is Block
    assert copied._data is block._data
    copied.set_feature_masks({'var_0': (1., 4.)})
    np.testing.assert_array_equal(copied.valid_windows, [[1, 2, 3]])
    np.testing.assert_array_equal(block.valid_windows, before)
    np.testing.assert_array_equal(copied.data, np.arange(5.)[:, None])


def test_summary_qualified_feature() -> None:
    """Qualified feature selectors must resolve to xarray variable names."""
    values = np.arange(10., dtype=float).reshape(5, 2)
    block = Block(**_block_kwargs(values))
    summary = block.summary(['var_1@0'])
    np.testing.assert_array_equal(summary.features.values, ['var_1'])


def test_cleanup_and_collection() -> None:
    """Cleanup releases arrays and timer factories do not retain blocks."""
    import gc
    import weakref

    block = Block(**_block_kwargs(np.arange(5.)[:, None]))
    _ = block.coords, block.valid_windows, block.fast_invalid_check
    timer = block.benchmark
    ref   = weakref.ref(block)
    block.cleanup()
    assert not {'coords', 'valid_windows', 'valid_mask'} & block.__dict__.keys()
    del block
    gc.collect()
    assert ref() is None
    assert callable(timer)


@pytest.mark.speed
def test_extraction_benchmark(record_property: Callable) -> None:
    """Record extraction timing without hardware-specific pass thresholds."""
    from time import perf_counter

    block = Block(**_block_kwargs(np.ones((128, 128, 4))))
    matches = np.arange(256)
    block.extract(matches, ['var_0'])
    start = perf_counter()
    for _ in range(10): block.extract(matches, ['var_0'])
    record_property('seconds_per_extract', (perf_counter() - start) / 10)


@pytest.mark.parametrize('optimize', [False, True])
def test_datafile_constructs_dense_block(optimize: bool) -> None:
    """The actual lazy Datafile factory produces the default dense subtype."""
    from crest.data.loading import Datafile

    values = np.arange(20., dtype=float).reshape(5, 4)
    source = Datafile(xr.Dataset(**{
        'data_vars' : {'v': (['x', 'y'], da.from_array(values))},
        'coords'    : {'x': np.arange(5.), 'y': np.arange(4.)},
    }), **{
        'features'     : ['v'],
        'window_depth' : {'x': 0, 'y': 0},
    })

    # Supply the same prepared summary used by a cached source, without I/O.
    source.summary = xr.DataArray(da.from_array([[0.]]), **{
        'dims'   : ['features', 'statistics'],
        'coords' : {'features': ['v'], 'statistics': ['null']},
    })
    source.update_blocks([1, 1])
    deferred = list(source.apply_overlap(0, optimize=optimize))
    block    = deferred[0]()
    assert type(block) is Block
    if not optimize: assert isinstance(block.sparsity, da.Array)
    output, _ = block.extract(np.array([0, 1]), ['v'])
    np.testing.assert_array_equal(output[:, 0, 0, 0], [0., 1.])


@pytest.mark.parametrize('fast', [False, True])
def test_extraction_does_not_construct_xarray(
    monkeypatch: pytest.MonkeyPatch, fast: bool,
) -> None:
    """Both output paths operate on raw arrays, never Sample/xarray views."""
    block = Block(**_block_kwargs(np.ones((3, 4, 2))))

    def unexpected(*args, **kwargs) -> None:
        raise AssertionError('Extraction must not construct xarray objects')

    monkeypatch.setattr(xr, 'Dataset', unexpected)
    monkeypatch.setattr(xr, 'DataArray', unexpected)
    output = block.extract(np.array([0, 1]), ['var_0'] if fast else None)
    assert len(output) == 2


@pytest.mark.parametrize('multichunk', [False, True])
@pytest.mark.parametrize('scheduler', ['synchronous', 'threads'])
def test_requested_features_share_source_read(
    multichunk: bool, scheduler: str,
) -> None:
    """Selected features share one source execution in each extraction call."""
    import dask

    calls  = []
    values = np.arange(60.).reshape(5, 4, 3)

    @dask.delayed(pure=False)
    def read_source() -> np.ndarray:
        calls.append(1)
        return values

    data = da.from_delayed(read_source(), **{
        'shape' : values.shape,
        'dtype' : values.dtype,
    })
    if multichunk: data = data.rechunk((2, 2, 1))
    data = [data[..., i:i+1] for i in range(3)]
    block = Block(**_block_kwargs(values, data=data))
    with dask.config.set(scheduler=scheduler):
        block.extract(np.array([0, 1]), ['var_0', 'var_2'])
    assert len(calls) == 1


@pytest.mark.parametrize('fast', [False, True])
def test_empty_feature_and_virtual_only_output(fast: bool) -> None:
    """Feature-only sources can omit every virtual coordinate dimension."""
    block = Block(**_block_kwargs(np.arange(3.).reshape(1, 1, 3), **{
        'original_dims' : (['features'], ['a', 'b', 'c'],
                           dict.fromkeys(['a', 'b', 'c'], float)),
    }))
    output = block.extract(np.array([0]), [] if fast else None)
    if fast:
        assert output[0].shape == (1, 0)
        assert output[1] == {}
    else:
        assert list(output[0][0]['coords']) == ['features']
        np.testing.assert_array_equal(output[0][0]['data'], [0., 1., 2.])


def test_multichunk_windows_cull_unselected_data() -> None:
    """A selected window reads neither other spatial nor feature chunks."""
    import dask

    @dask.delayed(pure=False)
    def unexpected() -> np.ndarray:
        raise AssertionError('Unselected source chunk was computed')

    values = np.arange(32.).reshape(4, 4, 2)
    bad    = da.from_delayed(unexpected(), shape=(2, 4, 1), dtype=float)
    good   = da.from_array(values[:2, :, :1], chunks=-1)
    first  = da.concatenate([good, bad], axis=0)
    other  = da.concatenate([bad, bad], axis=0)
    data   = da.concatenate([first, other], axis=-1)
    block  = Block(**_block_kwargs(values, data=data))
    output, _ = block.extract(np.array([0, 1]), ['var_0'])
    np.testing.assert_array_equal(output[:, 0, 0, 0], [0., 2.])


@pytest.mark.parametrize('dtype', [np.float32, np.int16, np.datetime64])
def test_raw_extraction_restores_feature_dtypes(dtype: type) -> None:
    """Raw windows restore each feature dtype without casting entire sources."""
    values = np.arange(8.).reshape(4, 2)
    dtype  = np.dtype('datetime64[m]') if dtype is np.datetime64 else dtype
    kwargs = _block_kwargs(values)
    kwargs['original_dims'][2]['var_1'] = dtype
    block = Block(**kwargs)
    output = block.extract(np.array([0, 1]))
    expected = values[:2, 1].astype(dtype)
    actual = np.array([group[0]['data'][1][0] for group in output])
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('percent', [0., 0.3, 1.])
def test_bin_masks_preserve_values_and_center_rules(percent: float) -> None:
    """Partial windows may contain excluded values, but not excluded centers."""
    values = np.array([1., 1., 1., 10., 1., 1., 1.])[:, None]
    block = Block(**_block_kwargs(values, **{
        'window_depth'  : {'x': (1, 1)},
        'valid_percent' : {('x',): percent},
    }))
    graphs = tuple(block._data)
    block.set_feature_masks({'var_0': (0., 2.)})
    centers = [1, 5] if percent == 1 else [1, 2, 4, 5]
    np.testing.assert_array_equal(block.valid_windows, [centers])
    np.testing.assert_array_equal(block.data, values)
    assert all(a is b for a, b in zip(graphs, block._data))
    assert block.valid_mask.shape == values.shape
    assert 'data' not in block.__dict__
    if percent < 1:
        output, _ = block.extract(np.array([1]), ['var_0'])
        np.testing.assert_array_equal(output[0, 0], [1., 1., 10.])


@pytest.mark.parametrize('percent', [0., 0.4, 1.])
def test_combined_bin_masks_preserve_update_semantics(percent: float) -> None:
    """Combine bins once while retaining cumulative window-mask behavior."""
    values = np.column_stack([np.arange(7.), np.arange(7.) ** 2])
    block = Block(**_block_kwargs(values, **{
        'valid_percent' : {('x',): percent},
    }))
    original = block._valid_mask
    block.set_feature_masks({'var_0': (1., 6.), 'var_1': (4., 26.)})
    expected = np.array([False, False, True, True, True, True, False])
    np.testing.assert_array_equal(block.valid_mask[..., 0], expected)
    np.testing.assert_array_equal(block._feature_mask.compute()[..., 0],
                                  expected)
    np.testing.assert_array_equal(original.compute(), np.ones((7, 1), bool))
    assert block._valid_mask.shape == (7, 1)

    # An unrelated source removes the current center mask, not prior validity.
    block.set_feature_masks({'var_0@1': (0., 7.)})
    assert not hasattr(block, '_feature_mask')
    np.testing.assert_array_equal(block.valid_mask[..., 0], expected)
    centers = list(range(7)) if percent == 0 else [2, 3, 4, 5]
    np.testing.assert_array_equal(block.valid_windows, [centers])


@pytest.mark.parametrize('lazy_vectors', [False, True])
def test_valid_coords_reads_only_vectors(lazy_vectors: bool) -> None:
    """Matching coordinates need neither a coordinate grid nor feature reads."""
    import dask

    @dask.delayed(pure=False)
    def forbidden() -> np.ndarray:
        raise AssertionError('Valid-center lookup read an unrelated array')

    values = np.ones((5, 4, 1), dtype='float32')
    vectors = [np.arange(5, dtype='float32') ** 2,
               np.array([np.nan, 2., 7., 11.], dtype='float32')]
    kwargs = _block_kwargs(values, **{
        'data'   : [da.from_delayed(forbidden(), **{
            'shape' : values.shape,
            'dtype' : values.dtype,
        })],
        'coords' : da.from_delayed(forbidden(), **{
            'shape' : (5, 4, 2),
            'dtype' : values.dtype,
        }),
        'coord_vecs' : [da.from_array(v, chunks=2) for v in vectors]
                      if lazy_vectors else vectors,
    })
    block = Block(**kwargs)
    indices = np.array([[4, 1, 4, 0], [2, 0, 2, 3]], dtype=np.uint16)
    block.valid_windows = indices
    expected = np.stack([v[i] for v, i in zip(vectors, indices)], axis=-1)
    np.testing.assert_array_equal(block.valid_coords, expected)
    assert block.valid_coords.dtype == np.dtype('float32')
    assert 'coords' not in block.__dict__
    assert 'data' not in block.__dict__


@pytest.mark.parametrize('lazy_vectors', [False, True])
def test_empty_valid_coords_without_reads(lazy_vectors: bool) -> None:
    """No valid centers returns correctly typed empty rows without computing."""
    import dask

    @dask.delayed(pure=False)
    def forbidden() -> np.ndarray:
        raise AssertionError('Empty coordinate lookup triggered a read')

    values = np.ones((3, 1), dtype='float32')
    vector = (da.from_delayed(forbidden(), shape=(3,), dtype='float32')
              if lazy_vectors else np.arange(3, dtype='float32'))
    block = Block(**_block_kwargs(values, **{
        'coords' : da.from_delayed(forbidden(), shape=(3, 1), dtype='float32'),
        'coord_vecs' : [vector],
    }))
    block.valid_windows = np.empty((1, 0), dtype=np.uint16)
    output = block.valid_coords
    assert output.shape == (0, 1)
    assert output.dtype == np.dtype('float32')
    assert 'coords' not in block.__dict__


@pytest.mark.parametrize('percent', [0., 0.4, 1.])
@pytest.mark.parametrize('boundary', [
    np.nan, 0., 'nearest', 'reflect', 'periodic',
    pytest.param('none', marks=pytest.mark.xfail(**{
        'raises' : AssertionError,
        'strict' : True,
        'reason' : 'Existing boundary=none block/mask shape mismatch',
    })),
])
def test_datafile_valid_coords_match_padded_grid(
    percent: float,
    boundary: float | str,
) -> None:
    """Vector lookup agrees with grid coordinates at every inbound center."""
    from crest.data.loading import Datafile

    values = np.arange(63., dtype=float).reshape(9, 7)
    values[3, 2] = np.nan
    source = Datafile(xr.Dataset(**{
        'data_vars' : {'v': (['x', 'y'], da.from_array(values, chunks=(5, 4)))},
        'coords'    : {'x': np.arange(9.) ** 2, 'y': np.arange(7.) / 2},
    }), **{
        'features'     : ['v'],
        'window_depth' : {'x': (1, 0), 'y': (0, 1)},
        'valid_percent' : {('x', 'y'): percent},
    })
    source.summary = xr.DataArray(da.from_array([[1.]]), **{
        'dims'   : ['features', 'statistics'],
        'coords' : {'features': ['v'], 'statistics': ['null']},
    })
    source.update_blocks([2, 2])
    for create in source.apply_overlap(1, boundary=boundary, optimize=False):
        block = create()
        grid  = block._coords.compute()
        expected = grid[tuple(block.valid_windows)]
        np.testing.assert_array_equal(block.valid_coords, expected)
        assert block.valid_coords.dtype == grid.dtype
        assert 'coords' not in block.__dict__


@pytest.mark.parametrize('boundary', [np.nan, 'nearest'])
def test_raw_window_padding_still_uses_coordinate_grid(
    boundary: float | str,
) -> None:
    """Matching uses vectors; raw coordinate windows retain grid padding."""
    from crest.data.loading import Datafile

    values = np.arange(36., dtype=float).reshape(6, 6)
    source = Datafile(xr.Dataset(**{
        'data_vars' : {'v': (['x', 'y'], da.from_array(values, chunks=(3, 3)))},
        'coords'    : {'x': np.arange(6.) + 10, 'y': np.arange(6.) + 20},
    }), **{
        'features'     : ['v'],
        'window_depth' : {'x': (1, 0), 'y': (1, 0)},
        'valid_percent' : {('x', 'y'): 0.},
    })
    source.summary = xr.DataArray(da.from_array([[0.]]), **{
        'dims'   : ['features', 'statistics'],
        'coords' : {'features': ['v'], 'statistics': ['null']},
    })
    source.update_blocks([2, 2])
    block = next(source.apply_overlap(1, boundary=boundary, optimize=False))()
    grid = block._coords.compute()
    centers = block.valid_windows.copy()
    _ = block.valid_coords
    count = centers.shape[1]
    output = block.extract(np.arange(count))
    for center, group in zip(centers.T, output):
        x, y = center
        expected = grid[x-1:x+1, y-1:y+1]
        np.testing.assert_array_equal(group[0]['coords']['x'],
                                      expected[:, 0, 0])
        np.testing.assert_array_equal(group[0]['coords']['y'],
                                      expected[0, :, 1])


@pytest.mark.parametrize('scheduler', ['synchronous', 'threads'])
def test_single_chunk_valid_coords_bounded_tasks(scheduler: str) -> None:
    """Repeated coordinate indices must not expand into per-slice Dask tasks."""
    import dask
    from dask.callbacks import Callback

    size = 128
    axis = da.arange(size, dtype='float32', chunks=-1)
    block = Block(**_block_kwargs(np.ones((size, size, 1)), **{
        'coord_vecs' : [axis, axis],
    }))
    block.valid_windows = np.indices((size, size)).reshape(2, -1)
    tasks = []
    with dask.config.set(scheduler=scheduler), Callback(**{
        'pretask' : lambda *args: tasks.append(args[0]),
    }):
        output = block.valid_coords
    np.testing.assert_array_equal(output, block.valid_windows.T)
    assert len(tasks) < 20
    assert 'coords' not in block.__dict__


@pytest.mark.parametrize('scheduler', ['synchronous', 'threads'])
def test_multichunk_valid_coords_cull_unselected_reads(scheduler: str) -> None:
    """Sparse selections still avoid reading unselected coordinate chunks."""
    import dask

    @dask.delayed(pure=False)
    def forbidden() -> np.ndarray:
        raise AssertionError('Unselected coordinate chunk was computed')

    good = da.from_array(np.array([5., 10.]), chunks=-1)
    bad  = da.from_delayed(forbidden(), shape=(2,), dtype=float)
    vector = da.concatenate([good, bad])
    block = Block(**_block_kwargs(np.ones((4, 1)), **{
        'coord_vecs' : [vector],
    }))
    block.valid_windows = np.array([[1, 0, 1]])
    with dask.config.set(scheduler=scheduler):
        output = block.valid_coords
    np.testing.assert_array_equal(output, [[10.], [5.], [10.]])


@pytest.mark.parametrize('scheduler', ['synchronous', 'threads'])
def test_coordinate_only_extraction_reads_selected_component(
    scheduler: str,
) -> None:
    """Selecting one coordinate must not read other components or data."""
    import dask

    @dask.delayed(pure=False)
    def forbidden() -> np.ndarray:
        raise AssertionError('Unrequested feature or coordinate was computed')

    values = np.ones((5, 4, 1))
    good = np.broadcast_to(np.arange(5.)[:, None, None], values.shape)
    bad = da.from_delayed(forbidden(), shape=values.shape, dtype=float)
    coords = da.concatenate([da.from_array(good, chunks=-1), bad], axis=-1)
    block = Block(**_block_kwargs(values, **{
        'data'   : [bad],
        'coords' : coords,
    }))
    with dask.config.set(scheduler=scheduler):
        output, _ = block.extract(np.array([0, 5]), ['x'])
    np.testing.assert_array_equal(output[:, 0, 0, 0], [0., 1.])


def test_combined_rule_followed_by_repeated_axis() -> None:
    """Intermediate edge validity must not remove valid interior centers."""
    block = Block(**_block_kwargs(np.ones((7, 7, 1)), **{
        'window_depth'  : {'x': (1, 1), 'y': (2, 2)},
        'valid_percent' : {('x', 'y'): 0.5, ('y',): 1.},
    }))

    expected = np.mgrid[1:6, 2:5].reshape(2, -1)
    np.testing.assert_array_equal(block.valid_windows, expected)