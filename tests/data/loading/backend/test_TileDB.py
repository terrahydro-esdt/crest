"""Regression tests for the TileDB sparse-read segfault (TileDB PR 5723).

Every sparse read starts a tile-merge task on the TileDB compute pool. The
last thing that task does is free a small temporary through the memory
resource of its query, but nothing waits for it: when the reading thread
releases the query first, the task frees through memory that no longer
exists. See https://github.com/TileDB-Inc/TileDB/pull/5723

Normally the task wins that race, and losing it only faults if the released
memory was already overwritten. Both tests therefore read in a child process
arranged so that the task loses on practically every read, and every loss is
fatal:

- the process is pinned to one CPU and the TileDB pool threads are demoted
  to SCHED_IDLE, so a pool thread is displaced the moment it wakes the
  reading thread and stays suspended until that thread blocks again;
- glibc fills freed memory with a byte pattern, so the late free always
  dereferences an invalid pointer.

These conditions change scheduling and memory contents only. They cannot
cause a fault in a library which does not use memory after freeing it.

"""
from __future__ import annotations
from crest.data.loading.backend.TileDB import _load_sparse_block
from crest.data.loading.backend.TileDB import tiledb_to_xarray

from pathlib import Path

import multiprocessing as mp
import tiledb as tdb
import faulthandler
import numpy as np
import resource
import signal
import pytest
import sys
import os


# Thread demotion and freed-memory filling rely on Linux and glibc
pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux only')

# Layout of the generated dataset: two blocks along the first dimension
_DIMS  = ('datetime', 'latitude', 'longitude')
_SHAPE = (8, 4, 4)
_TILE  = 4

# Reads per test, and the seconds after which a reader is abandoned. An
# unmitigated reader is expected to fault on its second read
_READS   = 100
_TIMEOUT = 60


def _create_group(path: str) -> np.ndarray:
    """Create a small TileDB group in the layout written by `TileDB.cache`.

    The group holds a dense coordinate array for each dimension, plus a
    sparse array named 'feature' which stores every other cell across four
    fragments.

    Parameters
    ----------
    path : str
        Location at which to create the TileDB group.

    Returns
    -------
    np.ndarray
        Dense copy of the feature values, with NaN for cells not stored.

    """

    tdb.group_create(path)
    with tdb.Group(path, mode='w') as group:

        # Coordinates have a single dimension with the same name
        dims = []
        for name, size in zip(_DIMS, _SHAPE):
            dims.append(tdb.Dim(**{
                'name'   : name,
                'domain' : (0, size-1),
                'tile'   : _TILE,
                'dtype'  : np.int64,
            }))
            schema = tdb.ArraySchema(**{
                'domain' : tdb.Domain(dims[-1]),
                'attrs'  : [tdb.Attr(name='data', dtype=np.int64)],
                'sparse' : False,
            })
            tdb.Array.create(f'{path}/{name}', schema)
            with tdb.open(f'{path}/{name}', mode='w') as A:
                A[:] = np.arange(size)
            group.add(name, name, relative=True)

        schema = tdb.ArraySchema(**{
            'domain' : tdb.Domain(*dims),
            'attrs'  : [tdb.Attr(name='data', dtype=np.float64)],
            'sparse' : True,
        })
        tdb.Array.create(f'{path}/feature', schema)

        # Each write creates one fragment; interleave their cells in space
        dense = np.full(_SHAPE, np.nan)
        cells = np.arange(0, dense.size, 2)
        with tdb.open(f'{path}/feature', mode='w') as A:
            for part in (cells[i::4] for i in range(4)):
                A[np.unravel_index(part, _SHAPE)] = part.astype(np.float64)
        dense.flat[cells] = cells
        group.add('feature', 'feature', relative=True)
    return dense


def _stress(path: str, mitigated: bool) -> None:
    """Read the feature repeatedly, under conditions which expose the race.

    This is the entry point of the child process: a segfault is a possible
    (or even the expected) outcome, and must not take pytest down with it.

    Parameters
    ----------
    path : str
        Location at which to create the TileDB group.
    mitigated : bool
        If True, read through `tiledb_to_xarray`, which applies the config
        workaround. Otherwise, call `_load_sparse_block` directly on an
        array which is opened without the workaround.

    """

    # A crash should leave a Python traceback, but not a core file
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    faulthandler.enable()

    # On a single CPU, a thread that wakes the reading thread is displaced
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    dense   = _create_group(path)
    threads = set(os.listdir('/proc/self/task'))

    if mitigated:
        feature = tiledb_to_xarray(path)['feature']
        load    = lambda: feature.compute().data

    # Use the same settings as tiledb_to_xarray, except for the workaround
    else:
        uri  = f'{path}/feature'
        data = tdb.open(uri, ctx=tdb.Ctx({
            'sm.compute_concurrency_level' : 1,
            'sm.io_concurrency_level'      : 1,
        }))
        info = {None: {
            'array-location' : [(0, size) for size in _SHAPE],
            'chunk-shape'    : _SHAPE,
        }}
        load = lambda: _load_sparse_block(uri, data, info)

    # Demote the TileDB threads started above, so that they only run while
    # the reading thread is blocked. The merge task of one read then stays
    # suspended at its final free until the next read has released the query
    for thread in set(os.listdir('/proc/self/task')) - threads:
        os.sched_setscheduler(int(thread), os.SCHED_IDLE, os.sched_param(0))

    for _ in range(_READS):
        np.testing.assert_array_equal(load().todense(), dense)


def _exitcode(
    path        : Path,
    mitigated   : bool,
    monkeypatch : pytest.MonkeyPatch,
) -> int:
    """Run `_stress` in a new interpreter and return its exit code.

    Parameters
    ----------
    path : Path
        Empty temporary directory, in which the dataset is created.
    mitigated : bool
        Whether to read through the mitigated interface (see `_stress`).
    monkeypatch : pytest.MonkeyPatch
        Fixture used to set the allocator options inherited by the child.

    Returns
    -------
    int
        Exit code of the child: 0 on success, or the negative signal number
        if it was killed by a signal (which includes exceeding `_TIMEOUT`).

    """

    # Have glibc fill freed memory, so that a use after free cannot pass
    # silently. The thread cache is disabled since cached frees are not filled
    monkeypatch.setenv('GLIBC_TUNABLES', ':'.join([
        'glibc.malloc.tcache_count=0',
        'glibc.malloc.perturb=165',
    ]))
    process = mp.get_context('spawn').Process(**{
        'target' : _stress,
        'args'   : (str(path / 'group.tiledb'), mitigated),
    })
    process.start()
    process.join(_TIMEOUT)

    # Has no effect on a child which already exited
    process.kill()
    process.join()
    return process.exitcode


def test_unmitigated_block_read_segfaults(
    tmp_path    : Path,
    monkeypatch : pytest.MonkeyPatch,
) -> None:
    """Reading through `_load_sparse_block` alone is expected to segfault.

    This test passes for as long as the installed TileDB contains the race.
    Once it fails with exit code 0, a TileDB release with the fix has been
    installed: the `preprocess_tile_merge` workaround in `tiledb_to_xarray`
    can then be removed, together with this test.

    """
    code = _exitcode(tmp_path, False, monkeypatch)
    assert(code == -signal.SIGSEGV), (
        f'Expected a segfault, but the reader exited with {code=} on TileDB '
        f'{tdb.libtiledb.version()}. If the code is 0, the installed TileDB '
        'no longer has the race: remove the preprocess_tile_merge workaround '
        'from tiledb_to_xarray, and this test. Any other code is a failure '
        'of the test itself; see the captured output.')


def test_mitigated_dataset_read_succeeds(
    tmp_path    : Path,
    monkeypatch : pytest.MonkeyPatch,
) -> None:
    """Reading the same data through `tiledb_to_xarray` must not segfault.

    The conditions are identical to those of the unmitigated test, so this
    passing while the other segfaults shows the workaround is what prevents
    the fault. The loaded values are also compared to the values written.

    """
    code = _exitcode(tmp_path, True, monkeypatch)
    assert(code == 0), (
        f'Mitigated reads failed with {code=} ({-signal.SIGSEGV} is a '
        'segfault); see the captured output for the traceback.')
