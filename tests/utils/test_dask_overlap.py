from __future__ import annotations

import numpy as np
import pytest

import dask.array as da
from dask.array.overlap import overlap as reference_overlap

from crest.utils.dask_overlap import (
    dask_overlap, 
    _add_checkpoint,
    _dependency_closure,
)


def _compute(array, scheduler='synchronous'):
    kwargs = {'scheduler': scheduler}
    if scheduler == 'threads':
        kwargs['num_workers'] = 4
    return array.compute(**kwargs)


def _assert_matches_reference(
    x,
    *,
    depth,
    boundary,
    allow_rechunk=False,
    blocks_per_batch=2,
    split_every=4,
    scheduler='synchronous',
):
    actual = dask_overlap(
        x,
        depth,
        boundary,
        allow_rechunk=allow_rechunk,
        blocks_per_batch=blocks_per_batch,
        split_every=split_every,
    )
    expected = reference_overlap(
        x,
        depth,
        boundary,
        allow_rechunk=allow_rechunk,
    )

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.chunks == expected.chunks
    assert actual.numblocks == expected.numblocks
    np.testing.assert_array_equal(
        _compute(actual, scheduler),
        _compute(expected, scheduler),
    )


@pytest.mark.parametrize(
    ('boundary', 'expected'),
    [
        ('none', [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7]),
        (
            -1,
            [-1, -1, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7, -1, -1],
        ),
        (
            'nearest',
            [0, 0, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7, 7, 7],
        ),
        (
            'reflect',
            [1, 0, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7, 7, 6],
        ),
        (
            'periodic',
            [6, 7, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7, 0, 1],
        ),
    ],
)
def test_1d_boundaries_have_explicit_expected_values(boundary, expected):
    x = da.from_array(np.arange(8), chunks=4)
    result = dask_overlap(
        x,
        depth=2,
        boundary=boundary,
        blocks_per_batch=1,
        split_every=2,
    )
    np.testing.assert_array_equal(_compute(result), expected)


@pytest.mark.parametrize(
    ('shape', 'chunks', 'depth', 'boundary'),
    [
        (
            (7, 8),
            ((3, 4), (3, 3, 2)),
            {0: 1, 1: 2},
            {0: 'none', 1: 'periodic'},
        ),
        (
            (7, 8),
            ((3, 4), (3, 3, 2)),
            {0: 2, 1: 1},
            {0: 'reflect', 1: -7},
        ),
        (
            (6, 7, 2),
            ((2, 2, 2), (3, 4), (1, 1)),
            {0: 1, 1: 2, 2: 0},
            {0: 'periodic', 1: 0, 2: 'none'},
        ),
    ],
)
def test_multidimensional_values_and_chunks_match_dask(
    shape,
    chunks,
    depth,
    boundary,
):
    values = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    x = da.from_array(values, chunks=chunks)
    _assert_matches_reference(
        x,
        depth=depth,
        boundary=boundary,
        blocks_per_batch=2,
    )


@pytest.mark.parametrize(
    ('blocks_per_batch', 'split_every'),
    [(1, 2), (2, 3), (5, 8), (12, 2), (100, 64)],
)
def test_batching_parameters_do_not_change_result(
    blocks_per_batch,
    split_every,
):
    values = np.arange(9 * 10).reshape(9, 10)
    x = da.from_array(values, chunks=((3, 3, 3), (4, 4, 2)))
    _assert_matches_reference(
        x,
        depth={0: 1, 1: 1},
        boundary={0: 'nearest', 1: 'periodic'},
        blocks_per_batch=blocks_per_batch,
        split_every=split_every,
    )


@pytest.mark.parametrize('scheduler', ['synchronous', 'threads'])
def test_stack_rechunk_map_graph_has_no_missing_dependencies(scheduler):
    """Regression coverage for the prior stack/rechunk failures."""
    values = np.arange(8 * 9).reshape(8, 9)
    base = da.from_array(values, chunks=(4, 3)).rechunk((3, 4))
    mapped = base.map_blocks(
        lambda block: block.astype(np.float32) * 1.5 + 2,
        dtype=np.float32,
    )
    x = da.stack(
        [mapped, da.zeros_like(mapped), mapped + 100],
        axis=-1,
    ).rechunk((3, 4, 1))

    _assert_matches_reference(
        x,
        depth={0: 1, 1: 1, 2: 0},
        boundary={0: 'nearest', 1: 'periodic', 2: 'none'},
        blocks_per_batch=2,
        scheduler=scheduler,
    )


def test_complex_graph_can_be_persisted_with_local_threads():
    base = da.from_array(
        np.arange(6 * 8).reshape(6, 8),
        chunks=(2, 4),
    )
    x = da.stack([base, base + 1], axis=-1).rechunk((2, 4, 1))
    kwargs = dict(
        depth={0: 1, 1: 1, 2: 0},
        boundary={0: 'reflect', 1: 0, 2: 'none'},
        allow_rechunk=False,
    )

    actual = dask_overlap(
        x,
        **kwargs,
        blocks_per_batch=2,
    ).persist(scheduler='threads', num_workers=4)
    expected = reference_overlap(x, **kwargs)

    np.testing.assert_array_equal(
        actual.compute(scheduler='synchronous'),
        expected.compute(scheduler='synchronous'),
    )


def test_zero_depth_returns_original_array():
    x = da.arange(12, chunks=3)
    assert dask_overlap(x, 0, 'none') is x
    assert dask_overlap(x, {0: 0}, {0: 'periodic'}) is x


def test_depth_larger_than_smallest_chunk_requires_rechunk_permission():
    x = da.from_array(np.arange(9), chunks=((4, 4, 1),))

    with pytest.raises(ValueError, match='exceeds the smallest chunk size'):
        dask_overlap(
            x,
            depth=2,
            boundary='nearest',
            allow_rechunk=False,
        )

    _assert_matches_reference(
        x,
        depth=2,
        boundary='nearest',
        allow_rechunk=True,
        blocks_per_batch=2,
    )


@pytest.mark.parametrize(
    ('kwargs', 'error', 'message'),
    [
        ({'blocks_per_batch': 0}, ValueError, 'blocks_per_batch'),
        ({'blocks_per_batch': -1}, ValueError, 'blocks_per_batch'),
        ({'split_every': 1}, ValueError, 'split_every'),
        ({'depth': {0: (1, 2)}}, NotImplementedError, 'symmetric'),
        ({'depth': {0: -1}}, ValueError, 'cannot be negative'),
    ],
)
def test_parameter_validation(kwargs, error, message):
    x = da.arange(12, chunks=4)
    call = dict(depth=1, boundary='none')
    call.update(kwargs)
    with pytest.raises(error, match=message):
        dask_overlap(x, **call)


def test_input_must_be_dask_array():
    with pytest.raises(TypeError, match='dask.array.Array'):
        dask_overlap(np.arange(8), 1, 'none')


def test_unsupported_boundary_fails_during_execution():
    x = da.arange(8, chunks=4)
    result = dask_overlap(
        x,
        depth=1,
        boundary='unsupported-boundary',
        blocks_per_batch=1,
    )
    with pytest.raises(ValueError, match='Unsupported boundary specification'):
        _compute(result)


def test_single_batch_creates_no_checkpoint_or_trim(monkeypatch):
    x = da.arange(8, chunks=2)
    result = dask_overlap(
        x,
        depth=1,
        boundary='periodic',
        blocks_per_batch=100,
        trim_every=1,
    )

    _compute(result)
    assert not any('checkpoint' in str(key) for key in result.dask)


def test_checkpoint_tree_has_expected_shape():
    tasks = {}
    output_keys = [('output', index) for index in range(10)]
    final_key = _add_checkpoint(
        tasks,
        output_keys,
        name='checkpoint',
        split_every=3,
    )

    # 10 map tasks + 4 first-level reductions + 2 second-level reductions
    # + 1 final reduction.
    assert len(tasks) == 17
    assert final_key == ('checkpoint', 'done')
    assert sum(key[1] == 'reduce' for key in tasks) == 6


def test_dependency_closure_is_transitive_and_rejects_missing_keys():
    graph = {key: object() for key in ('source', 'middle', 'output', 'other')}
    dependencies = {
        'source': frozenset(),
        'middle': frozenset({'source'}),
        'output': frozenset({'middle'}),
        'other': frozenset(),
    }

    assert _dependency_closure(
        {'output'},
        graph,
        dependencies,
    ) == {'source', 'middle', 'output'}

    with pytest.raises(ValueError, match='missing from the materialized'):
        _dependency_closure(
            {'missing'},
            graph,
            dependencies,
        )
