# CREST logging

## LoggingService

```python
from crest.utils.logging_service import LoggingService, get_logger

# Initialize logging (call once at application startup)
LoggingService.initialize()

# Get a logger for your module
logger = get_logger(__name__)

# Use the logger
logger.info("application started")
logger.debug("detailed debug information")
logger.warning("warning")
logger.error("error occurred")
```

## Backward compatibility

```python
from crest.utils.crest_logger import logger_setup
import logging

# Initialize logging
logger_setup()

# Get a logger
logger = logging.getLogger(__name__)
logger.info("Application started")
```

**Note:** `LoggingService.initialize()` calls `logger_setup()` internally, so they provide the same core functionality.

## Where handlers attach, and coexisting with other handlers

By default (`scope="root"`) CREST attaches its handlers - application
log file, error log file, console - to the Python **root logger**, so any
module's `logging.getLogger(__name__)` is captured, whatever package it
belongs to. Pass `scope="crest"` (or set `CREST_LOG_SCOPE=crest`) to attach
them only to the `"crest"` logger instead, isolated with `propagate=False`.
The `"crest.metrics"` logger does not propagate; it writes to its own
metrics file (or, under loadEnv, to the run file - see below).

The root logger is shared, so other code may already have handlers on it:
a host application's own setup, a shell environment's run log, pytest's
log capture (`caplog`), or a library that called `logging.basicConfig()`.
CREST tags every handler it creates and only ever removes its own:

- `clear_handlers=True` (default): repeated setup -
  `logger_setup()`, `LoggingService.initialize(force=True)`,
  `LoggingService.reconfigure()` - replaces CREST's previous handlers and
  leaves everyone else's attached, in their original order. Removed
  handlers are closed, so long-running processes don't leak file
  descriptors.
- `clear_handlers=False`: nothing is removed; new handlers are added
  alongside the existing ones.
- `clear_handlers="all"`: every handler on the target logger is removed and
  closed, whoever installed it. Use this only when CREST should take over
  the root logger completely.

`LoggingService.reset()` follows the same rule: it removes CREST's handlers
and restores the levels and `propagate` flags CREST changed, and leaves the
rest alone.

**Duplicate console output.** CREST does not deduplicate *output*. If the
root logger already has a console handler (for example from
`logging.basicConfig()`) and CREST is configured with `console=True`, every
record is printed by both. Either pass `console=False`, or remove the other
handler, or use `clear_handlers="all"` if CREST should own the console.

**`logging.basicConfig()` does nothing once CREST is configured.** This is
standard Python behavior: `basicConfig()` only acts when the root logger has no
handlers, and under the default `scope="root"` CREST's handlers are on it. So:

- Don't mix the two: configure logging with CREST alone. Its `console`,
  `console_level` and `format_type` options cover what `basicConfig()` is
  usually used for.
- If a script must call `basicConfig()`, call it *before* CREST and pass
  `console=False` to CREST, or every record is printed twice (see above).
- `basicConfig(force=True)` removes **and closes** every root handler,
  including CREST's. A plain `LoggingService.initialize()` afterwards does
  nothing, because CREST still counts as initialized. Use
  `LoggingService.initialize(force=True)` to put CREST's handlers back.

**CREST never raises the root logger's level.** Under `scope="root"` it sets
the root level to the lower of what it was before CREST first configured it
and `log_level`, and applies `log_level` through its own handlers. So
`log_level="WARNING"` keeps INFO out of CREST's files, but does not hide INFO
records from a handler someone else attached to root at INFO.
(`LoggingService.reset()` puts the root level back to what it was.)

Libraries - including CREST's own modules - should only call
`logging.getLogger(__name__)` and leave handlers and levels to the
application that configures logging.

```python
import logging
from crest.utils.logging_service import LoggingService

# A handler someone else installed on the root logger...
run_log = logging.FileHandler("run.log")
logging.getLogger().addHandler(run_log)

LoggingService.initialize()                  # ...is still attached
LoggingService.initialize(force=True)        # ...and still attached once
LoggingService.reset()                       # ...and still attached
```

## Running under loadEnv (ast_utils)

Processes started from a loadEnv shell whose profile has a `logging` block get
automatic logging from ast_utils (`etc/lib/loadenv_logging.py`, started by its
`sitecustomize.py`): a run ID, one JSON **run file** per process in
`CREST_LOG_DIR`, and a run index that `runlog` reads. CREST is built to sit on
top of that without duplicating it.

**Who does what**

| loadEnv (ast_utils) | CREST |
|---|---|
| The run file on the root logger, one per process (`terrahydro-<pid>.log.jsonl`) | The console |
| `CREST_RUN_ID`, the run index, `runlog` | Levels: `log_level`, `console_level`, `module_levels` |
| Stamping `run_id`, `stage`, `host`, `pid`, Slurm IDs on every record | Metrics (`"crest.metrics"`), `log_context()`, `install_global_exception_logger()` |
| `print()` capture (`CREST_LOG_STDIO=1`) | Its own files, only when there's no run log (or `run_log="ignore"`) |

Nothing to configure beyond the usual single call:

```python
from crest.utils.logging_service import LoggingService, get_logger

LoggingService.initialize()
get_logger(__name__).info("written once, to the run file")
```

**Each record is written to disk once.** With the default `run_log="auto"`,
when loadEnv's run file is on the root logger CREST doesn't create its
application or error log files (records already reach the run file through the
root logger; `runlog --level ERROR` gives the errors-only view). Metrics, which
don't propagate, are sent to the run file too, under the logger name
`crest.metrics`, instead of `metrics.jsonl`. CREST borrows that handler: it
never closes it, and `reset()` only detaches it from `"crest.metrics"`.
`run_log="ignore"` makes CREST write its own files as well.

`run_log="auto"` only applies under `scope="root"`. Under `scope="crest"`,
CREST's records don't propagate to the root logger, so they never reach the run
file, and CREST keeps writing its own files.

**Don'ts**

- Don't use `clear_handlers="all"`: it removes and closes loadEnv's run file
  handler too. (CREST then falls back to its own files.)
- Don't export `CREST_LOG_SCOPE=crest` in a loadEnv profile: every process
  would stop sending CREST's records to the run file. Pass `scope` in code if a
  process needs it.
- Don't set `run_id` yourself, in `log_context()` or `extra=`. It is loadEnv's
  process run ID; a different value takes the record out of that run in
  `runlog`. Label runs or experiments with `run` instead:

```python
from crest.utils.logging_service import get_logger, log_context

logger = get_logger(__name__)
with log_context(run="exp_001", stage="training"):
    logger.info("epoch 1")                                  # run="exp_001", stage="training"
    logger.info("fetching", extra={"stage": "download"})    # extra wins: stage="download"
```

`log_context()` fields win over loadEnv's stamps (e.g. `stage` from
`CREST_STAGE`), and `extra=` wins over both. Contexts are per thread and per
asyncio task.

**Things to know**

- `logging.basicConfig()`: loadEnv patches it to step aside for the run file,
  but once CREST is configured it is still a no-op (see above).
- The run file's level is the profile's `level` (`CREST_LOG_LEVEL`). Metrics
  are logged at INFO, so with a profile level above INFO they don't reach the
  run file.
- With `CREST_LOG_STDIO=1` (batch jobs), loadEnv copies what's printed to
  stderr into the run file as `crest.stdio`, skipping lines written by
  `logging` handlers, so CREST's console isn't copied. Older ast_utils
  versions only recognized a handler writing to loadEnv's stream *directly*.
  Importing CREST's model code imports `prettytable` (and `numba`), which wrap
  `sys.stderr` with colorama, so with those versions CREST's console lines
  reach the run file twice (as the record and as `crest.stdio`). Update
  ast_utils, or in batch jobs pass `console=False`, or call
  `colorama.deinit()` after your imports and before
  `LoggingService.initialize()`.
- With `CREST_LOG_STDIO=1` and CREST's `install_global_exception_logger()`, an
  uncaught exception's traceback is in the run file twice: once as captured
  stderr (`crest.stdio`, from Python's default hook, which CREST's hook still
  calls) and once as CREST's `CRITICAL` record.
- Interactive sessions (a Python prompt, IPython, Jupyter) are skipped by
  loadEnv, so there is no run log and no `CREST_RUN_ID`. CREST then writes its
  own files to `CREST_LOG_DIR`, naming the application file from
  `CREST_LOG_FILE` (so it's per process). `error.jsonl` and `metrics.jsonl`
  keep fixed names, shared by every process on the host, unless you give them
  a `{pid}` (below).
- Text format (`CREST_LOG_FORMAT=text`) lines start with the same ISO
  timestamp as loadEnv's, but the rest of the layout differs. `runlog --text`
  prints its own layout from the JSON files.

### Per-process files and record fields outside loadEnv

The same conventions apply when there's no loadEnv, so CREST's own files read
like run files:

- File names may contain `{pid}` and `{host}` (short host name), so processes
  sharing a folder don't share files. Any other placeholder raises
  `ValueError`; use `{{`/`}}` for literal braces.
- When `log_file` is left at its default and there's no run log, CREST uses
  `CREST_LOG_FILE` if set (loadEnv exports `terrahydro-{pid}.log.jsonl`).
- Timestamps are ISO 8601 with milliseconds, in `CREST_LOG_TZ`
  (default `America/New_York`; local time if the zone name is invalid).
- JSON records include `host` and `pid`, and `run_id`, `stage`,
  `slurm_job_id`, `slurm_array_task_id` when `CREST_RUN_ID`, `CREST_STAGE`,
  `SLURM_JOB_ID`, `SLURM_ARRAY_TASK_ID` are set, unless the record already has
  them.

```python
from crest.utils.logging_service import LoggingService

LoggingService.initialize(
    log_dir="logs",
    log_file="app-{host}-{pid}.log.jsonl",
    error_file="app-{host}-{pid}.error.jsonl",
    metrics_file="metrics-{pid}.jsonl",
)
```

Any file output can be switched off with `None`, e.g. console only:

```python
from crest.utils.logging_service import LoggingService

LoggingService.initialize(log_file=None, error_file=None, metrics_file=None)
```

`metrics_file=None` drops metrics records (also under loadEnv). CREST only
creates `log_dir` when it writes a file there.

## Examples

### Development: use verbose logging

```python
from crest.utils.logging_service import LoggingService

# Enable DEBUG level for development
LoggingService.initialize(log_level="DEBUG", format_type="text")
```

### Production: use "quiet console"

```python
from crest.utils.logging_service import LoggingService

# Log everything to file, but only warnings/errors to console
LoggingService.initialize(
    log_level="INFO",
    console_level="WARNING"
)
```

### Log Levels

Use string names, integers, or the `LogLevel` enum:

```python
from crest.utils.logging_service import LoggingService
from crest.utils.crest_logger import LogLevel

# Set either way:
LoggingService.initialize(log_level="DEBUG")
LoggingService.initialize(log_level=10)  # logging.DEBUG
LoggingService.initialize(log_level=LogLevel.DEBUG)
```

Available levels (from most to least verbose):
- `DEBUG` (10): Detailed diagnostic information
- `INFO` (20): General informational messages
- `WARNING` (30): Warning messages
- `ERROR` (40): Error messages
- `CRITICAL` (50): Critical errors

## Environment Variables

Configure logging via environment variables:

```bash
# Set default log level
export CREST_LOG_LEVEL=DEBUG

# Set console log level (can differ from file)
export CREST_CONSOLE_LOG_LEVEL=WARNING

# Set log directory
export CREST_LOG_DIR=/s3/thdro/crest

# Set format
export CREST_LOG_FORMAT=text

# Application log file name when log_file is left at its default
# (ignored when loadEnv's run log is present)
export CREST_LOG_FILE='app-{pid}.log.jsonl'

# Time zone of timestamps (default America/New_York)
export CREST_LOG_TZ=UTC
```

`CREST_RUN_ID`, `CREST_STAGE`, `SLURM_JOB_ID` and `SLURM_ARRAY_TASK_ID`, when
set, are added to JSON records (see "Running under loadEnv"). Parameters always
win over environment variables.

Then:
```python
from crest.utils.logging_service import LoggingService

LoggingService.initialize()  # Will use environment variables
```

Or with the alternative API:
```python
from crest.utils.crest_logger import logger_setup

logger_setup()  # Will also use environment variables
```

## Enhancements

### Log rotation

Prevent log files from growing unbounded with automatic rotation:

#### Size-based rotation

```python
from crest.utils.logging_service import LoggingService

# Rotate when log reaches 10MB, keep 5 backups
LoggingService.initialize(
    rotate=True,
    max_bytes=10 * 1024 * 1024,  # 10MB
    backup_count=5
)
```

#### Time-based rotation

```python
from crest.utils.logging_service import LoggingService

# Rotate daily at midnight, keep 30 days
LoggingService.initialize(
    rotate=True,
    rotate_when='midnight',
    backup_count=30
)

# Rotate every hour, keep 24 hours
LoggingService.initialize(
    rotate=True,
    rotate_when='H',
    backup_count=24
)

# Rotate every Monday, keep 4 weeks
LoggingService.initialize(
    rotate=True,
    rotate_when='W0',  # W0=Monday, W6=Sunday
    backup_count=4
)
```

**Time-based rotation options:**

- `'S'` - Seconds
- `'M'` - Minutes
- `'H'` - Hours
- `'D'` - Days
- `'midnight'` - Daily at midnight
- `'W0'` through `'W6'` - Weekly (Monday=W0, Sunday=W6)

### Per-module configuration

Control verbosity for specific modules:

```python
from crest.utils.logging_service import LoggingService

LoggingService.initialize(
    log_level="INFO",  # Default for all modules
    module_levels={
        "crest.model": "DEBUG",           # Verbose model logs
        "crest.data.batching": "DEBUG",   # Verbose batching logs
        "crest.archiver": "WARNING",      # Quiet archiver logs
        "crest.data_server": "ERROR"      # Very quiet data server
    }
)
```

## Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_file` | str \| None | `"crest.log.jsonl"` | Application log filename (`{pid}`/`{host}` allowed; `None`: no file) |
| `error_file` | str \| None | `"error.jsonl"` | Errors-only log filename (same rules) |
| `metrics_file` | str \| None | `"metrics.jsonl"` | Metrics log filename (same rules; `None` drops metrics) |
| `console` | bool | `True` | Enable console output |
| `log_level` | str \| int \| `LogLevel` | `CREST_LOG_LEVEL` or `"INFO"` | Level for CREST's files; root level is only ever lowered |
| `console_level` | str \| int \| `LogLevel` | `CREST_CONSOLE_LOG_LEVEL` or `log_level` | Level for the console |
| `scope` | str | `"root"` | Handler target: `"root"` or `"crest"` (see above) |
| `log_dir` | str | `"logs"` | Directory for log files |
| `format_type` | str | `"json"` | Format: "json" or "text" |
| `module_levels` | dict | `None` | Per-module log levels |
| `clear_handlers` | bool \| `"all"` | `True` | Remove CREST's previous handlers (`True`), nothing (`False`), or every handler (`"all"`) |
| `rotate` | bool | `False` | Enable log rotation |
| `max_bytes` | int | `10485760` (10MB) | Max log size before rotation |
| `backup_count` | int | `5` | Number of backup files to keep |
| `rotate_when` | str | `None` | Time-based rotation interval |
| `run_log` | `"auto"` \| `"ignore"` | `"auto"` | Under loadEnv: skip CREST's app/error files and send metrics to the run file (`"auto"`), or write CREST's files anyway (`"ignore"`) |


## References

- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
