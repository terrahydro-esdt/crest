# Changelog

All notable changes to CREST will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-11-11

### Summary

Fixes sporadic segfaults in Batcher worker processes reading TileDB data, adds
sparse blocks and an xvec backend to the data layer, and reworks logging so
CREST cooperates with the host application's handlers and with loadEnv's run
logs. Logging behavior changed in several places; see Changed before upgrading.

### Changed

#### Data and models

- `GriddedModel` prediction post-processing keeps the raw prediction output as
  `<output>.backup` instead of deleting it, so it needs about twice the disk
  space. Delete the backup once the post-processed output has been checked.
- Caching a `Datafile` raises `RuntimeError` when its `datetime` coordinate
  is a datetime type other than `datetime64[ns]` (for example
  `datetime64[us]` after resampling with pandas 3), instead of writing a cache
  that fails later.
- Caching sparse data to TileDB checks fill values: `valid_mask` must be
  boolean with fill value `False`, every other feature must have fill value
  `NaN`; anything else raises `ValueError`. `valid_mask` is now written to
  the cache too.
- Without directed sampling, `MultiBatcher` skips a block when its block
  configuration already has more than `max_queue` batches queued, so batches
  no longer pile up for common configurations while it waits on rare ones.
- A Batcher writes its initialization parameters at the top of its log file.
- `CITATION.cff` and the README cite the Zenodo concept DOI
  (`10.5281/zenodo.21939176`), which always resolves to the latest release.

#### Logging

- `logger_setup()`/`LoggingService.initialize()` now default to `scope="root"`
  - configured handlers attach to the Python root logger
  instead of the `"crest"` logger, so any caller's own
  `logging.getLogger(__name__)` is captured automatically, without needing
  to name its logger `"crest"` or route through `get_logger()`. Pass
  `scope="crest"` to keep the previous behavior (handlers attached only to
  `"crest"`, isolated via `propagate=False`).
- `get_logger()` / `LoggingService.get_logger()` no longer silently rewrites
  a caller's logger name to add a `"crest."` prefix. The returned logger's
  name always matches what was requested (or the detected caller
  `__name__`).
- Reconfiguring logging (`clear_handlers=True`, the default;
  `LoggingService.initialize(force=True)`; `LoggingService.reconfigure()`)
  now removes only the handlers CREST itself installed. Handlers added by
  the host application, by pytest's log capture, or by another library
  (e.g. a run-log file handler on the root logger) stay attached. Pass
  `clear_handlers="all"` to remove every handler on the target logger, as
  `clear_handlers=True` did under `scope="root"` before this change.
- Under `scope="root"`, CREST no longer raises the root logger's level. It
  sets root to the lower of its original level and `log_level`, and applies
  `log_level` on its own handlers, so a quieter `log_level` (e.g.
  `"WARNING"`) no longer hides INFO records from handlers CREST doesn't own.
- Under loadEnv (ast_utils' run-file handler on the root logger), CREST no
  longer writes `crest.log.jsonl` or `error.jsonl` by default, and metrics go
  to the run file instead of `metrics.jsonl`: every record is written to
  disk once. Use `runlog --level ERROR` for an errors-only view, or pass
  `run_log="ignore"` to keep CREST's own files.
- JSON and text timestamps are ISO 8601 with millisecond (was microsecond)
  precision, in the `CREST_LOG_TZ` zone (default `America/New_York`, as
  before). Text lines previously used `%Y-%m-%d %H:%M:%S` in local time.
- JSON records now include `host` and `pid`, and `run_id`, `stage`,
  `slurm_job_id`, `slurm_array_task_id` when the matching environment
  variables are set (see Added). Key names of existing fields are unchanged.
- `log_context()` fields are now per thread and per asyncio task, instead of
  applying to records from every thread while the block runs. Code that
  relied on another thread picking up a context loses those fields.
- `log_dir` is only created when CREST writes a file there.

### Fixed

#### Data and models

- Batcher worker processes reading TileDB sparse arrays crashed sporadically
  with a segfault, caused by a race condition in TileDB's sparse global-order
  reader ([TileDB #5723](https://github.com/TileDB-Inc/TileDB/pull/5723)).
  CREST now switches off the affected code path
  (`sm.query.sparse_global_order.preprocess_tile_merge = 0`), and the per-block
  read lock that only made the crash rarer is gone. A regression test
  reproduces the crash; it is expected to fail (xfail) on TileDB releases known
  to have the bug (tiledb-py 0.36.1 / libtiledb 2.30.1 and earlier). Once a
  fixed TileDB is released, upgrade and remove the workaround.
- A Batcher could hang forever when a worker process died before any batch
  was produced, because the queues between processes become unusable. It now
  logs an error and shuts down.
- Re-queueing blocks between epochs could deadlock on a full block queue. The
  queue is larger, a full queue logs a warning instead of blocking, and with
  `epoch_sync` the other workers wait until the first one has re-filled it.
- `Datafile` sparsity of an entirely empty sparse block was a 0-d
  `sparse.COO` object instead of a number, which broke sorting blocks by
  sparsity. It is now `1.0`.
- `Datafile` only sorts numeric coordinates, so non-numeric coordinates (such
  as xvec geometries) no longer break loading.
- `Dataset`'s check that blocks are large enough for the sample window applied
  `valid_percent` to the wrong side of the comparison. The error now also
  shows both sizes.
- Selecting a dimension coordinate as a sample feature returns that sample's
  coordinate values, with `NaN` (or `NaT` for `datetime`) where a sample has
  none.
- The brute-force coordinate matcher (`multiset_full_numba`) undercounted
  pending task sizes, which could lead to slower task orderings.
- `Node` no longer removes a size-1 dimension that its expected input shape
  leaves unspecified (`None`).
- Comparison plots written to TensorBoard have the batch dimension
  `tf.summary.image` expects, use log-scaled axes when the values span three
  or more orders of magnitude, and come back empty instead of failing when
  there is nothing to plot.

#### Logging

- `LoggingService.reset()` only ever cleared handlers/level from
  `"crest"`-prefixed loggers, never the root logger, so it silently failed
  to undo initialization once `scope="root"` became the default. It now
  removes (and closes) every handler CREST installed - on the root logger,
  `"crest"`, `"crest.metrics"` and any `module_levels` logger - and
  restores the levels and `propagate` flags CREST changed, leaving
  handlers and settings it did not install alone.
- Handlers removed during reconfiguration were never closed, leaking one
  set of log-file descriptors per reconfiguration in long-running
  processes. Removed handlers are now closed.
- `LoggingService.initialize()` never forwarded `error_file`, `rotate`,
  `max_bytes`, `backup_count`, or `rotate_when` to `logger_setup()`. 
   All five are now accepted and forwarded.
- Logging with `extra={"stage": ...}` (or any key) inside a
  `log_context()` that sets the same key raised
  `KeyError: "Attempt to overwrite 'stage' in LogRecord"`. The per-call value
  now wins, and context fields still win over values loadEnv stamps from the
  environment.

### Added

#### Data and models

- `SparseBlock` (`crest.data.loading`): a `Block` for sparse data, such as
  point observations stored in TileDB. `Datafile` picks it automatically when
  the data's chunks are `sparse.COO`. Note that support for windows over
  sparse dimensions is still limited; see the `SparseBlock` docstring for
  details.
- `Xvec` backend for xvec-encoded Zarr stores (paths ending in `.xvec`), for
  data on vector geometries such as polygons. Needs the optional `xvec`
  package.
- `clip` transform (`Transform.clip`), limiting values to a range (default
  `[-6, 6]`). It can't be inverted.
- `cicd/environment_cpu_windows.yaml`, a conda environment for Windows (CPU).
- Releases are cut with `scripts/release.py` and published to GitHub (and
  archived on Zenodo) by a tag-triggered workflow; see
  `docs/DEV_PROD_WORKFLOW.md`.

#### Logging

- `logger_setup()`/`LoggingService.initialize()` accept a `scope` parameter
  (`"root"` default, `"crest"` opt-in) and a matching `CREST_LOG_SCOPE`
  environment variable fallback.
- `clear_handlers` accepts `"all"` in addition to `True`/`False`; any
  other value raises `ValueError`.
- `run_log="auto"|"ignore"` on `logger_setup()`/`LoggingService.initialize()`:
  whether CREST defers file output to loadEnv's run log when one is present
  (`"auto"`, default, and only under `scope="root"`). Any other value raises
  `ValueError`.
- `log_file=None` and `metrics_file=None` (like `error_file=None`) switch
  those outputs off; `metrics_file=None` drops metrics records.
- `{pid}` and `{host}` placeholders in `log_file`, `error_file` and
  `metrics_file`, for per-process files in a shared folder. Other
  placeholders raise `ValueError` naming the parameter.
- Environment variables: `CREST_LOG_FILE` (application file name when
  `log_file` is left at its default and there's no run log), `CREST_LOG_TZ`
  (time zone of timestamps), and `CREST_RUN_ID`, `CREST_STAGE`,
  `SLURM_JOB_ID`, `SLURM_ARRAY_TASK_ID` (added to JSON records unless
  already present).

## [0.1.0] - 2026-08-14

### Summary

First release.

### Fixed

- `GriddedModel` no longer hardcodes a filesystem path as the default `SysMetrics` 
  run directory; falls back to `SysMetrics`'s own default (`logs/<timestamp>`).
- `pyproject.toml`'s `docs` extra was missing packages `docs/conf.py` actually
  requires (`myst-nb`, `nbsphinx`, `nbsphinx-link`, `pydata-sphinx-theme`,
  `sphinxcontrib-mermaid`), so `pip install ".[docs]"` alone could not build the
  documentation.

[Unreleased]: https://github.com/terrahydro-esdt/crest/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/terrahydro-esdt/crest/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/terrahydro-esdt/crest/releases/tag/v0.1.0
