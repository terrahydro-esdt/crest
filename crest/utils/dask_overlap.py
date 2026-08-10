"""
Provides a replacement for dask.overlap.overlap that:
  - removes forced rechunking
  - enables control of heap memory trimming for faster eviction
  - significantly reduces the memory footprint necessary for overlaps
"""

from collections.abc import Iterable
from numbers import Integral
from typing import Any

from dask._task_spec import (
    List as TaskList,
    Task,
    TaskRef,
    convert_legacy_graph,
)
from dask.array.overlap import (
    coerce_boundary,
    coerce_depth,
    ensure_minimum_chunksize,
)
from dask.base import clone_key, tokenize
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from dask.utils import ensure_dict

import dask.array as da
import numpy as np
import functools
import itertools
import ctypes
import ctypes.util


@functools.lru_cache(maxsize=1)
def _get_malloc_trim():
    """Return the process-local glibc malloc_trim function."""
    try:
        libc_name = ctypes.util.find_library('c') or 'libc.so.6'
        libc = ctypes.CDLL(libc_name, use_errno=True)
    except FileNotFoundError:
        return None, None
        
    try:
        malloc_trim = libc.malloc_trim
    except AttributeError as exc:
        return None, None
        raise RuntimeError(
            'malloc_trim is unavailable; this likely is not a glibc system.'
        ) from exc

    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int

    # Keep both alive
    return libc, malloc_trim


def _trim_heap_after(_checkpoint_result):
    """
    Trim free glibc heap pages after a checkpoint.

    Returns None so this remains a lightweight graph dependency.
    """
    _, malloc_trim = _get_malloc_trim()
    if malloc_trim is not None: 
        malloc_trim(0)
    return None

    
def _is_boundary(boundary: Any, value: str) -> bool:
    """Safely compare a boundary specification with a string."""
    return isinstance(boundary, str) and boundary == value


def _after(value, _blocker):
    """Return value, but only after blocker has completed."""
    return value


def _checkpoint_none(*_values):
    """Consume dependencies and replace their results with None."""
    return None


def _batched(iterable: Iterable, size: int):
    """Yield tuples containing at most `size` items."""
    iterator = iter(iterable)

    while batch := tuple(itertools.islice(iterator, size)):
        yield batch


def _dependency_closure(
    required_keys,
    graph,
    internal_dependencies,
):
    """
    Return all graph keys needed to calculate `required_keys`.

    `internal_dependencies` contains only dependencies that are themselves
    present in `graph`.
    """
    closure = set()
    pending = list(required_keys)

    while pending:
        key = pending.pop()

        if key in closure:
            continue

        if key not in graph:
            raise ValueError(
                f'Required source key {key!r} is missing from the '
                'materialized source graph.'
            )

        closure.add(key)
        pending.extend(internal_dependencies[key])

    return closure


def _add_checkpoint(
    tasks,
    output_keys,
    *,
    name,
    split_every=8,
    trim=False,
):
    """
    Add a memory-light checkpoint over `output_keys`.

    One map task consumes each output independently and returns None.
    The resulting None values are reduced through a small tree.

    Returns
    -------
    Hashable
        The final checkpoint key.
    """
    split_every = int(split_every)

    if split_every < 2:
        raise ValueError('split_every must be at least 2.')

    current = []

    # Map each potentially large array result to None independently. This
    # allows the array result to be released without waiting for every other
    # output in the batch to become available
    for position, output_key in enumerate(output_keys):
        checkpoint_key = (name, 'map', position)

        tasks[checkpoint_key] = Task(
            checkpoint_key,
            _checkpoint_none,
            TaskRef(output_key),
        )

        current.append(checkpoint_key)

    if not current:
        raise ValueError('Cannot create a checkpoint over zero output keys.')

    level = 0

    while len(current) > split_every:
        next_level = []

        for position, start in enumerate(
            range(0, len(current), split_every)
        ):
            group = current[start : start + split_every]
            reduction_key = (name, 'reduce', level, position)

            tasks[reduction_key] = Task(
                reduction_key,
                _checkpoint_none,
                *[TaskRef(key) for key in group],
            )

            next_level.append(reduction_key)

        current = next_level
        level += 1

    final_key = (name, 'done')

    tasks[final_key] = Task(
        final_key,
        _checkpoint_none,
        *[TaskRef(key) for key in current],
    )

    if trim:
        trim_key = (name, 'malloc-trim')
    
        tasks[trim_key] = Task(
            trim_key,
            _trim_heap_after,
            TaskRef(final_key),
        )
        return trim_key
    return final_key


def _neighbor_spec(
    index,
    *,
    depth,
    boundary,
    numblocks,
):
    """
    Return overlap offsets and corresponding source-block indices.

    Global non-periodic boundaries are omitted here and applied later with
    NumPy padding inside `_assemble_overlap_block`.
    """
    axis_choices = []

    for axis, block_index in enumerate(index):
        halo = depth[axis]
        kind = boundary[axis]
        axis_numblocks = numblocks[axis]

        choices = [(0, block_index)]

        if halo:
            if block_index > 0:
                choices.insert(0, (-1, block_index - 1))
            elif _is_boundary(kind, 'periodic'):
                choices.insert(0, (-1, axis_numblocks - 1))

            if block_index + 1 < axis_numblocks:
                choices.append((1, block_index + 1))
            elif _is_boundary(kind, 'periodic'):
                choices.append((1, 0))

        axis_choices.append(choices)

    combinations = tuple(itertools.product(*axis_choices))

    offsets = tuple(
        tuple(item[0] for item in combination)
        for combination in combinations
    )

    neighbor_indices = tuple(
        tuple(item[1] for item in combination)
        for combination in combinations
    )

    return offsets, neighbor_indices


def _overlap_chunks(
    chunks,
    *,
    depth,
    boundary,
):
    """Calculate chunk sizes produced by the overlap operation."""
    output_chunks = []

    for axis, axis_chunks in enumerate(chunks):
        halo = depth[axis]
        kind = boundary[axis]
        numblocks = len(axis_chunks)

        new_axis_chunks = []

        for block_index, chunk_size in enumerate(axis_chunks):
            has_left_halo = (
                block_index > 0
                or not _is_boundary(kind, 'none')
            )
            has_right_halo = (
                block_index + 1 < numblocks
                or not _is_boundary(kind, 'none')
            )

            new_axis_chunks.append(
                chunk_size
                + halo * int(has_left_halo)
                + halo * int(has_right_halo)
            )

        output_chunks.append(tuple(new_axis_chunks))

    return tuple(output_chunks)


def _assemble_overlap_block(
    blocks,
    offsets,
    depth,
    boundary,
    index,
    numblocks,
):
    """
    Assemble one overlapped NumPy block from neighboring source blocks.

    Parameters
    ----------
    blocks
        Resolved NumPy source blocks.
    offsets
        Relative block offset for each source block.
    depth
        Symmetric overlap depth along each axis.
    boundary
        Boundary condition along each axis.
    index
        Global output block index.
    numblocks
        Number of source blocks along each axis.
    """
    pieces = dict(zip(offsets, blocks))
    ndim = len(index)

    center_offset = (0,) * ndim
    center = pieces[center_offset]
    center_shape = center.shape

    # These widths represent actual neighboring blocks. Non-periodic global
    # boundary padding is applied afterward
    real_left_width = tuple(
        depth[axis]
        if (
            index[axis] > 0
            or _is_boundary(boundary[axis], 'periodic')
        )
        else 0
        for axis in range(ndim)
    )

    real_right_width = tuple(
        depth[axis]
        if (
            index[axis] + 1 < numblocks[axis]
            or _is_boundary(boundary[axis], 'periodic')
        )
        else 0
        for axis in range(ndim)
    )

    core_shape = tuple(
        real_left_width[axis]
        + center_shape[axis]
        + real_right_width[axis]
        for axis in range(ndim)
    )

    output = np.empty(core_shape, dtype=center.dtype)

    # Fill the center, faces, edges, and corners using neighboring blocks
    for offset, piece in pieces.items():
        source_slices = []
        destination_slices = []

        for axis, direction in enumerate(offset):
            halo = depth[axis]
            center_start = real_left_width[axis]
            center_size = center_shape[axis]

            if direction < 0:
                source_slices.append(
                    slice(piece.shape[axis] - halo, piece.shape[axis])
                )
                destination_slices.append(slice(0, halo))

            elif direction > 0:
                source_slices.append(slice(0, halo))

                destination_start = center_start + center_size
                destination_slices.append(
                    slice(destination_start, destination_start + halo)
                )

            else:
                source_slices.append(slice(None))
                destination_slices.append(
                    slice(center_start, center_start + center_size)
                )

        output[tuple(destination_slices)] = piece[
            tuple(source_slices)
        ]

    # Apply global non-periodic boundaries one axis at a time. This matches
    # Dask's boundary-application order
    for axis in range(ndim):
        halo = depth[axis]
        kind = boundary[axis]

        if (
            halo == 0
            or _is_boundary(kind, 'none')
            or _is_boundary(kind, 'periodic')
        ):
            continue

        left_pad = halo if index[axis] == 0 else 0
        right_pad = (
            halo
            if index[axis] + 1 == numblocks[axis]
            else 0
        )

        if not left_pad and not right_pad:
            continue

        pad_width = [(0, 0)] * ndim
        pad_width[axis] = (left_pad, right_pad)

        if _is_boundary(kind, 'nearest'):
            output = np.pad(
                output,
                pad_width,
                mode='edge',
            )

        elif _is_boundary(kind, 'reflect'):
            # Dask's reflect boundary includes the edge value. NumPy calls
            # this behavior 'symmetric'
            output = np.pad(
                output,
                pad_width,
                mode='symmetric',
            )

        elif isinstance(kind, str):
            raise ValueError(
                f'Unsupported boundary specification {kind!r} '
                f'on axis {axis}.'
            )

        else:
            output = np.pad(
                output,
                pad_width,
                mode='constant',
                constant_values=kind,
            )

    return output


def dask_overlap(
    x,
    depth,
    boundary,
    *,
    allow_rechunk=False,
    blocks_per_batch=8,
    split_every=8,
    trim_every=0,
):
    """
    Apply overlap with bounded upstream dependency lifetimes.

    Unlike Dask's standard overlap graph, source chunks are shared only within
    a small output-block batch. Batches are separated by explicit checkpoint
    dependencies, preventing source chunks from remaining live across the
    entire output array.

    Parameters
    ----------
    x : dask.array.Array
        Source array.
    depth : int, tuple, or dict
        Symmetric overlap depth. Asymmetric depth tuples are not supported by
        this implementation.
    boundary : str, scalar, or dict
        Per-axis boundary behavior: 'none', 'periodic', 'nearest', 'reflect',
        or a scalar constant.
    allow_rechunk : bool, default False
        Rechunk axes whose smallest chunk is smaller than the overlap depth.
    blocks_per_batch : int, default 8
        Number of output blocks calculated before enforcing a checkpoint.
        This is the primary peak-memory control.
    split_every : int, default 8
        Branching factor for checkpoint reduction tasks.

    Returns
    -------
    dask.array.Array
        An array with the same block grid as `x`, with each block expanded by
        its appropriate overlap.
    """

    if not isinstance(x, da.Array):
        raise TypeError(
            f'x must be a dask.array.Array, not {type(x).__name__}.'
        )

    blocks_per_batch = int(blocks_per_batch)

    if blocks_per_batch < 1:
        raise ValueError('blocks_per_batch must be at least 1.')

    split_every = int(split_every)

    if split_every < 2:
        raise ValueError('split_every must be at least 2.')

    coerced_depth = coerce_depth(x.ndim, depth)
    coerced_boundary = coerce_boundary(x.ndim, boundary)

    normalized_depth = []

    for axis in range(x.ndim):
        axis_depth = coerced_depth.get(axis, 0)

        if isinstance(axis_depth, tuple):
            if len(axis_depth) != 2:
                raise ValueError(
                    f'Invalid depth tuple {axis_depth!r} on axis {axis}.'
                )

            if axis_depth[0] != axis_depth[1]:
                raise NotImplementedError(
                    'This bounded overlap implementation supports only '
                    'symmetric overlap depths.'
                )

            axis_depth = axis_depth[0]

        if not isinstance(axis_depth, Integral):
            raise TypeError(
                f'Overlap depth on axis {axis} must be an integer, '
                f'not {type(axis_depth).__name__}.'
            )

        if axis_depth < 0:
            raise ValueError(
                f'Overlap depth on axis {axis} cannot be negative.'
            )

        normalized_depth.append(int(axis_depth))

    depth_tuple = tuple(normalized_depth)

    if not any(depth_tuple):
        return x

    boundary_tuple = tuple(
        coerced_boundary.get(axis, 'none')
        for axis in range(x.ndim)
    )

    # Ensure every chunk is wide enough to supply its requested halo
    rechunked_chunks = list(x.chunks)

    for axis, halo in enumerate(depth_tuple):
        if halo == 0:
            continue

        smallest_chunk = min(x.chunks[axis])

        if halo <= smallest_chunk:
            continue

        if not allow_rechunk:
            raise ValueError(
                f'Overlap depth {halo} on axis {axis} exceeds the '
                f'smallest chunk size {smallest_chunk}. Pass '
                'allow_rechunk=True or rechunk the array first.'
            )

        rechunked_chunks[axis] = ensure_minimum_chunksize(
            halo,
            x.chunks[axis],
        )

    rechunked_chunks = tuple(rechunked_chunks)

    if rechunked_chunks != x.chunks:
        x = x.rechunk(rechunked_chunks)

    output_chunks = _overlap_chunks(
        x.chunks,
        depth=depth_tuple,
        boundary=boundary_tuple,
    )

    token = tokenize(
        x,
        depth_tuple,
        boundary_tuple,
        blocks_per_batch,
        split_every,
    )
    output_name = f'bounded-overlap-{token}'

    # This is the only full materialization/conversion of the source graph
    source_graph = convert_legacy_graph(
        ensure_dict(x.__dask_graph__())
    )

    # Calculate dependencies once. External dependencies are deliberately
    # excluded from the closure because they are not keys within this graph
    internal_dependencies = {
        key: frozenset(
            dependency
            for dependency in node.dependencies
            if dependency in source_graph
        )
        for key, node in source_graph.items()
    }

    output_indices = np.ndindex(*x.numblocks)
    batches = list(_batched(output_indices, blocks_per_batch))

    tasks = {}
    previous_checkpoint = None

    for batch_number, batch_indices in enumerate(batches):
        output_specs = []
        required_source_keys = set()

        # Determine all source chunks needed by this output batch
        for output_index in batch_indices:
            offsets, neighbor_indices = _neighbor_spec(
                output_index,
                depth=depth_tuple,
                boundary=boundary_tuple,
                numblocks=x.numblocks,
            )

            source_keys = tuple(
                (x.name,) + neighbor_index
                for neighbor_index in neighbor_indices
            )

            required_source_keys.update(source_keys)

            output_specs.append(
                (
                    output_index,
                    offsets,
                    source_keys,
                )
            )

        closure = _dependency_closure(
            required_source_keys,
            source_graph,
            internal_dependencies,
        )

        batch_seed = (token, batch_number)

        substitutions = {
            old_key: clone_key(old_key, seed=batch_seed)
            for old_key in closure
        }

        # Clone the required upstream subgraph. Only frontier nodes are
        # attached to the previous checkpoint; all downstream tasks inherit
        # that dependency naturally
        for old_key in closure:
            new_key = substitutions[old_key]

            cloned_node = source_graph[old_key].substitute(
                substitutions,
                key=new_key,
            )

            if (
                previous_checkpoint is not None
                and not internal_dependencies[old_key]
            ):
                cloned_node = Task(
                    new_key,
                    _after,
                    cloned_node,
                    TaskRef(previous_checkpoint),
                )

            tasks[new_key] = cloned_node

        batch_output_keys = []

        # Create final overlap tasks directly under the final array name
        for output_index, offsets, source_keys in output_specs:
            output_key = (output_name,) + output_index

            input_blocks = TaskList(
                *[
                    TaskRef(substitutions[source_key])
                    for source_key in source_keys
                ]
            )

            tasks[output_key] = Task(
                output_key,
                _assemble_overlap_block,
                input_blocks,
                offsets,
                depth_tuple,
                boundary_tuple,
                output_index,
                x.numblocks,
            )

            batch_output_keys.append(output_key)

        # The final batch needs no checkpoint because nothing follows it
        if batch_number + 1 < len(batches):
            checkpoint_name = (
                f'{output_name}-checkpoint-{batch_number}'
            )
            should_trim = (
                (trim_every >= 1) and
                ((batch_number + 1) % int(trim_every) == 0)
            )
            previous_checkpoint = _add_checkpoint(
                tasks,
                batch_output_keys,
                name=checkpoint_name,
                split_every=split_every,
                trim=should_trim,
            )

    layer = MaterializedLayer(tasks)

    graph = HighLevelGraph(
        layers={output_name: layer},
        dependencies={output_name: set()},
    )

    return da.Array(
        graph,
        output_name,
        chunks=output_chunks,
        dtype=x.dtype,
        meta=x._meta,
    )
