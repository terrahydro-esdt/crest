"""
This module implements a multipurpose logging utility.
"""

from __future__ import annotations
import datetime as dt
import json
import logging
import os
import pathlib
import platform
import sys
import json
import io
import re
from zoneinfo import ZoneInfo
from enum import Enum
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Literal, Optional


class LogLevel(Enum):
    """Log level enumeration for type-safe level specification."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


def _team_time(created: float) -> dt.datetime:
    """`created` (a POSIX timestamp) as an aware datetime in the team's time zone.

    The zone is read from CREST_LOG_TZ on each call (default
    "America/New_York"), the same as loadEnv's run log, falling back to the
    machine's local time if the zone name isn't valid.
    """
    try:
        return dt.datetime.fromtimestamp(
            created, ZoneInfo(os.environ.get("CREST_LOG_TZ") or "America/New_York")
        )
    except Exception:
        return dt.datetime.fromtimestamp(created).astimezone()


# Record fields loadEnv's run log stamps from the environment (ast_utils
# loadenv_logging.RUN_FIELDS); CREST's JSON formatter adds the same ones.
_RUN_FIELDS_FROM_ENV = {
    "run_id": "CREST_RUN_ID",
    "stage": "CREST_STAGE",
    "slurm_job_id": "SLURM_JOB_ID",
    "slurm_array_task_id": "SLURM_ARRAY_TASK_ID",
}


class _IsoTextFormatter(logging.Formatter):
    """Text formatter whose ``%(asctime)s`` is the JSON formatter's timestamp:
    ISO 8601, milliseconds, in the CREST_LOG_TZ zone (like loadEnv's text format)."""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        if datefmt:
            return super().formatTime(record, datefmt)
        return _team_time(record.created).isoformat(timespec="milliseconds")


class CrestJSONFormatter(logging.Formatter):
    """Define a CREST JSON Formatter used by CREST loggers

    Allows for the storage of logs in JSON format so that they can be programmatically
    parsed. Every record carries the fields loadEnv's run log stamps: ``host`` (short
    host name) and ``pid``, plus ``run_id``, ``stage``, ``slurm_job_id`` and
    ``slurm_array_task_id`` when CREST_RUN_ID, CREST_STAGE, SLURM_JOB_ID and
    SLURM_ARRAY_TASK_ID are set. Values already on the record (stamped by loadEnv, set by
    ``log_context``, or passed in ``extra``) are kept.

    Parameters
    ----------
    fmt_keys: dict[str, str]
        a customizable formatter dictionary with a format key, i.e. that which will
        appear in the log message, and the value, which is the variable from the log
        record.
    """

    def __init__(self, *, fmt_keys: dict[str, str] | None = None):
        super().__init__()
        self.fmt_keys = fmt_keys or {}

    def format(self, record: logging.LogRecord) -> str:
        """ Dumps the log as a json """
        msg = self._prepare_log_dict(record)
        return json.dumps(msg, default=str, ensure_ascii=False)

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict[str, any]:
        """ Returns a dictionary of logged information """
        always = {
            "message": record.getMessage(),
            "timestamp": _team_time(record.created).isoformat(timespec="milliseconds"),
        }
        if record.exc_info is not None:
            always["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info is not None:
            always["stack_info"] = self.formatStack(record.stack_info)

        front = {
            key: (
                always.pop(val, None) if val in always else getattr(
                    record, val, None)
            )
            for key, val in self.fmt_keys.items()
        }
        out = {k: v for k, v in front.items() if v is not None}
        out.update(always)

        for k, v in record.__dict__.items():
            if k not in LOG_RECORD_BUILTIN_ATTRS and k not in out:
                out[k] = v

        # Identity fields, as in loadEnv's run log, so CREST's own files read the same way.
        out.setdefault("host", _short_host())
        out.setdefault("pid", record.process or os.getpid())
        for key, env_var in _RUN_FIELDS_FROM_ENV.items():
            if os.environ.get(env_var):
                out.setdefault(key, os.environ[env_var])
        return out


# Attribute set on every handler logger_setup() creates, so CREST can later
# tell its own handlers apart from ones a host application, test framework,
# or another library attached to the same logger.
_CREST_OWNED_ATTR = "_crest_owned"


def _attach_owned(logger: logging.Logger, handler: logging.Handler) -> None:
    """Mark `handler` as CREST-owned and attach it to `logger`."""
    setattr(handler, _CREST_OWNED_ATTR, True)
    logger.addHandler(handler)


def _remove_handlers(logger: logging.Logger, owned_only: bool = True) -> None:
    """Detach and close handlers on `logger`.

    With `owned_only=True`, only handlers CREST attached (see `_attach_owned`)
    are removed; handlers added by anyone else stay attached in their
    original order. With `owned_only=False`, every handler is removed.
    """
    for handler in list(logger.handlers):
        if owned_only and not getattr(handler, _CREST_OWNED_ATTR, False):
            continue
        logger.removeHandler(handler)
        handler.close()


# Handlers CREST attached but didn't create (loadEnv's run-file handler on
# "crest.metrics"), as (logger, handler) pairs. CREST detaches them on
# reconfiguration and reset but never closes them: their owner still uses them.
_borrowed_handlers: list[tuple[logging.Logger, logging.Handler]] = []


def _attach_borrowed(logger: logging.Logger, handler: logging.Handler) -> None:
    """Attach someone else's `handler` to `logger` (once), remembering to detach it later."""
    if handler not in logger.handlers:
        logger.addHandler(handler)
    if (logger, handler) not in _borrowed_handlers:
        _borrowed_handlers.append((logger, handler))


def _detach_borrowed(logger: Optional[logging.Logger] = None) -> None:
    """Detach, without closing, the borrowed handlers on `logger` (on every logger if None)."""
    for lg, handler in list(_borrowed_handlers):
        if logger is None or lg is logger:
            lg.removeHandler(handler)
            _borrowed_handlers.remove((lg, handler))


# Original (level, propagate) of every logger logger_setup() has changed,
# recorded the first time it changes each one, so that a reset can put the
# process back the way CREST found it.
_original_logger_state: dict[logging.Logger, tuple[int, bool]] = {}


def _record_original_state(logger: logging.Logger) -> None:
    """Remember `logger`'s level and propagate flag, unless already recorded."""
    _original_logger_state.setdefault(logger, (logger.level, logger.propagate))


def _reset_logging_state() -> None:
    """Undo what logger_setup() did to the process's loggers.

    Detaches (without closing) the handlers CREST borrowed, removes (and
    closes) every CREST-owned handler on the root logger and on every named
    logger, then restores the level and propagate flag recorded for each
    logger CREST changed. Handlers, levels and flags CREST did not install or
    change are left alone.
    """
    _detach_borrowed()
    loggers = [logging.getLogger()] + [
        lg for lg in logging.Logger.manager.loggerDict.values()
        if isinstance(lg, logging.Logger)
    ]
    for lg in loggers:
        _remove_handlers(lg, owned_only=True)

    for lg, (level, propagate) in _original_logger_state.items():
        lg.setLevel(level)
        lg.propagate = propagate
    _original_logger_state.clear()


def _find_run_log_handler() -> Optional[logging.Handler]:
    """Return loadEnv's run-file handler if it is attached to root, else None.

    ast_utils' `loadenv_logging` (started by its `sitecustomize`) keeps that
    handler in a module global. CREST only looks it up in `sys.modules` and
    never imports it, so there is no dependency on ast_utils. A public
    `run_file_handler()` accessor, if ast_utils adds one, is preferred over
    the private `_file_handler` global. Anything unexpected counts as "no run
    log", so CREST falls back to writing its own files.
    """
    try:
        module = sys.modules.get("loadenv_logging")
        if module is None:
            return None
        accessor = getattr(module, "run_file_handler", None)
        handler = accessor() if callable(accessor) else getattr(module, "_file_handler", None)
        if isinstance(handler, logging.Handler) and handler in logging.getLogger().handlers:
            return handler
    except Exception:
        pass
    return None


def _short_host() -> str:
    """This machine's host name up to the first dot, the same as loadEnv's ``{host}``."""
    return platform.node().split(".")[0] or "unknown"


def _format_file_name(param: str, name: Optional[str]) -> Optional[str]:
    """Fill the ``{pid}`` and ``{host}`` placeholders in a log file name.

    Raises ValueError, naming `param`, for any other placeholder or a stray
    brace, rather than creating a file literally called ``{foo}.jsonl``.
    """
    if not name:
        return name
    try:
        return name.format(pid=os.getpid(), host=_short_host())
    except (KeyError, IndexError, ValueError, AttributeError) as e:
        raise ValueError(
            f"Invalid {param}: {name!r}. Only the {{pid}} and {{host}} placeholders "
            f"are supported (use {{{{ and }}}} for literal braces)"
        ) from e


def logger_setup(
    log_file: Optional[str] = "crest.log.jsonl",
    metrics_file: Optional[str] = "metrics.jsonl",
    error_file: Optional[str] = "error.jsonl",
    console: bool = True,
    log_level: Optional[str | int | LogLevel] = None,
    console_level: Optional[str | int | LogLevel] = None,
    scope: Optional[Literal["root", "crest"]] = None,
    log_dir: str = "logs",
    format_type: str = "json",
    module_levels: Optional[dict[str, str | int]] = None,
    clear_handlers: bool | Literal["all"] = True,
    rotate: bool = False,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    rotate_when: Optional[str] = None,
    run_log: Literal["auto", "ignore"] = "auto",
) -> None:
    """
    This function sets up the logging infrastructure for CREST.

    Parameters
    ----------
    log_file : str or None, default="crest.log.jsonl"
        Name of the application log file. None writes no application log file.
        This and the other two file names may contain ``{pid}`` (the process
        ID) and ``{host}`` (the host name up to the first dot), so that
        processes sharing `log_dir` write separate files, e.g.
        "app-{host}-{pid}.log.jsonl". Any other placeholder raises ValueError.
    metrics_file : str or None, default="metrics.jsonl"
        Name of the metrics log file. None switches metrics off: records
        logged to "crest.metrics" are dropped, also under a run log.
    error_file : str or None, default="error.jsonl"
        Name of the errors log file. None writes no errors log file.
    console : bool, default=True
        Enable console output to stderr.
    log_level : str | int | LogLevel, optional
        Default log level for file output (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        If None, reads from CREST_LOG_LEVEL environment variable, defaults to INFO.
        Under scope="root" the root logger's level is set to min(original level, `log_level`)
        so a quieter `log_level` never hides records from handlers CREST doesn't own. 
        Under scope="crest" the "crest" logger is set to exactly `log_level`.
    console_level : str | int | LogLevel, optional
        Log level for console output (can differ from file level).
        If None, reads from CREST_CONSOLE_LOG_LEVEL environment variable,
        defaults to same as log_level.
    scope : "root" | "crest", optional
        Where configured handlers are attached:

        - "root" (default): handlers attach to the Python root logger, so
          *any* logger obtained via ``logging.getLogger(__name__)`` from
          any module in any package is captured automatically. This is the
          recommended mode for applications that embed CREST (e.g.
          terrahydro, terrahweb) and want their own logging to "just work"
          without special naming.
        - "crest": handlers attach only to the ``"crest"`` logger, which is
          set to ``propagate=False``. Only loggers under the ``crest.*``
          hierarchy are captured; logging from other packages is
          unaffected. Use this when a host application wants CREST's logs
          kept isolated from its own.

        If None, reads from the CREST_LOG_SCOPE environment variable,
        defaults to "root".
    log_dir : str, default="logs"
        Directory for log files (relative or absolute path).
    format_type : str, default="json"
        Log format: "json" for structured JSON logs, "text" for human-readable format.
    module_levels : dict[str, str | int], optional
        Per-module log levels for fine-grained control. Keys must be the
        real dotted module or package name of the logger being configured
        (whatever name that module passes to ``logging.getLogger(__name__)``)
        - not a scope label. This works the same way regardless of `scope`:
        Example: {"crest.model": "DEBUG", "terrahydro.data.ingestion": "WARNING"}
        A key that doesn't match any logger actually in use (e.g. a stale
        or invented name) silently has no effect, the same as calling
        ``logging.getLogger("some.unused.name").setLevel(...)`` would.
    clear_handlers : bool | "all", default=True
        Which existing handlers to remove (and close) before adding new ones:

        - True: only handlers a previous CREST setup installed, so repeated
          setup calls don't duplicate output. Handlers attached by the host
          application, a test framework (e.g. pytest's log capture), or
          another library are left in place.
        - False: remove nothing; new handlers are added alongside existing
          ones.
        - "all": remove every handler on the target logger and on
          "crest.metrics", whoever installed it. Use this only when the
          host wants CREST to take over the root logger completely.
    rotate : bool, default=False
        Enable log file rotation to prevent unbounded file growth.
    max_bytes : int, default=10485760 (10MB)
        Maximum size of log file before rotation (when rotate=True and rotate_when=None).
    backup_count : int, default=5
        Number of backup log files to keep when rotating.
    rotate_when : str, optional
        Time-based rotation interval: 'S' (seconds), 'M' (minutes), 'H' (hours),
        'D' (days), 'midnight', 'W0'-'W6' (weekday). If specified, uses time-based
        rotation instead of size-based.
    run_log : "auto" | "ignore", default="auto"
        What to do when the process already has a run log, i.e. loadEnv's
        run-file handler (ast_utils ``loadenv_logging``) is on the root logger:

        - "auto": don't create the application and error log files, whatever
          `log_file`/`error_file` say. Records reach the run file through the
          root logger, so writing them to CREST's files too would duplicate
          them. Metrics records (which never propagate) are sent to the run
          file too, instead of `metrics_file`. The console, levels and
          `module_levels` are unaffected.
        - "ignore": write CREST's files anyway, as if there were no run log.

        Only applies under scope="root"; under scope="crest" records don't
        propagate to the root logger, so CREST always writes its own files.

    Environment Variables
    ---------------------
    CREST_LOG_LEVEL : str
        Default log level if not specified in parameters (e.g., "DEBUG", "INFO")
    CREST_CONSOLE_LOG_LEVEL : str
        Console log level if not specified in parameters
    CREST_LOG_SCOPE : str
        Handler attachment scope ("root" or "crest") if not specified in
        parameters
    CREST_LOG_DIR : str
        Log directory if not specified in parameters
    CREST_LOG_FORMAT : str
        Log format ("json" or "text") if not specified in parameters
    CREST_LOG_TZ : str
        Time zone of JSON and text timestamps (default "America/New_York"; the
        machine's local time if the name isn't a valid zone)
    CREST_RUN_ID, CREST_STAGE, SLURM_JOB_ID, SLURM_ARRAY_TASK_ID : str
        Added to JSON records as ``run_id``, ``stage``, ``slurm_job_id`` and
        ``slurm_array_task_id`` unless the record already has them (as loadEnv does)
    CREST_LOG_FILE : str
        Application log file name (``{pid}``/``{host}`` allowed) if `log_file`
        is left at its default. Ignored when a run log is present, since
        loadEnv uses this variable to name its own run file.

    Examples
    --------
    Basic usage with defaults (INFO level, JSON format, console enabled):

    >>> from crest.utils.crest_logger import logger_setup
    >>> logger_setup()

    Enable DEBUG logging for development:

    >>> logger_setup(log_level="DEBUG")

    Quiet console, verbose file logging:

    >>> logger_setup(log_level="DEBUG", console_level="WARNING")

    Per-module configuration:

    >>> logger_setup(
    ...     log_level="INFO",
    ...     module_levels={
    ...         "crest.model": "DEBUG",
    ...         "crest.data.batching": "DEBUG",
    ...         "crest.archiver": "WARNING"
    ...     }
    ... )

    Human-readable text format for interactive development:

    >>> logger_setup(format_type="text", log_level="DEBUG")

    One application log file per process, for processes sharing a log folder:

    >>> logger_setup(log_file="app-{host}-{pid}.log.jsonl")  # doctest: +SKIP

    Keep CREST's logs isolated from a host application's own logging:

    >>> logger_setup(scope="crest")

    Using environment variables:

    >>> # export CREST_LOG_LEVEL=DEBUG
    >>> # export CREST_CONSOLE_LOG_LEVEL=WARNING
    >>> logger_setup()  # Will use env vars

    Notes
    -----
    - This function should be called once at application startup
    - Calling it multiple times will reconfigure logging (if clear_handlers=True),
      replacing only CREST's own handlers;
      prefer `LoggingService.initialize()` for idempotent repeated calls
    - With the default scope="root", handlers attach to the Python root
      logger, so any module's `logging.getLogger(__name__)` is captured -
      CREST places no special requirement on logger naming
    - With scope="crest", handlers attach only to the "crest" logger
      (propagate=False), isolating CREST's logs from a host application's
    - The "crest.metrics" logger is always configured separately and doesn't
      propagate; it logs to metrics_file, regardless of `scope`, except
      under a run log with run_log="auto", where it logs to the run file,
      and with metrics_file=None, where metrics are dropped
    - `log_dir` is only created when a file is written there
    - Under loadEnv (a run log on the root logger), the default
      run_log="auto" leaves file output for the application to the run log
    - JSON format is recommended for production/automated parsing
    - Text format is recommended for development/debugging
    """

    # Resolve log level from parameter or environment variable
    if log_level is None:
        log_level = os.getenv("CREST_LOG_LEVEL", "INFO").upper()
    log_level = _parse_log_level(log_level)

    # Resolve console level
    if console_level is None:
        env_console_level = os.getenv("CREST_CONSOLE_LOG_LEVEL")
        console_level = env_console_level.upper() if env_console_level else log_level
    console_level = _parse_log_level(console_level)

    # Resolve scope from parameter or environment variable
    if scope is None:
        scope = os.getenv("CREST_LOG_SCOPE", "root").lower()
    if scope not in ("root", "crest"):
        raise ValueError(f"Invalid scope: {scope!r}. Must be 'root' or 'crest'")

    if not (isinstance(clear_handlers, bool) or clear_handlers == "all"):
        raise ValueError(
            f"Invalid clear_handlers: {clear_handlers!r}. Must be True, False or 'all'"
        )

    if run_log not in ("auto", "ignore"):
        raise ValueError(f"Invalid run_log: {run_log!r}. Must be 'auto' or 'ignore'")

    # loadEnv exports CREST_LOG_FILE as its own run file's name. Use it only
    # when there's no run log; otherwise, under run_log="ignore", CREST would
    # open loadEnv's file a second time.
    log_file_param = "log_file"
    env_log_file = os.environ.get("CREST_LOG_FILE")
    if log_file == "crest.log.jsonl" and env_log_file and _find_run_log_handler() is None:
        log_file, log_file_param = env_log_file, "CREST_LOG_FILE"
    log_file = _format_file_name(log_file_param, log_file)
    error_file = _format_file_name("error_file", error_file)
    metrics_file = _format_file_name("metrics_file", metrics_file)

    # Resolve log directory
    if log_dir == "logs" and "CREST_LOG_DIR" in os.environ:
        log_dir = os.getenv("CREST_LOG_DIR")

    # Resolve format type
    if format_type == "json" and "CREST_LOG_FORMAT" in os.environ:
        format_type = os.getenv("CREST_LOG_FORMAT", "json").lower()

    logs_dir = pathlib.Path(log_dir)

    def file_handler(name: str) -> logging.FileHandler:
        """A handler on `logs_dir/name`, rotating as configured; creates `logs_dir`."""
        logs_dir.mkdir(parents=True, exist_ok=True)
        path = logs_dir / name
        if rotate and rotate_when:
            return TimedRotatingFileHandler(
                path, when=rotate_when, backupCount=backup_count, encoding="utf-8"
            )
        if rotate:
            return RotatingFileHandler(
                path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
        return logging.FileHandler(path, mode="a", encoding="utf-8")

    # Create formatter based on format_type
    if format_type == "json":
        formatter = CrestJSONFormatter(
            fmt_keys={
                "level": "levelname",
                "logger": "name",
                "kind": "kind",
                "stage": "stage",
                "run": "run",
            }
        )
    elif format_type == "text":
        formatter = _IsoTextFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        raise ValueError(
            f"Invalid format_type: {format_type}. Must be 'json' or 'text'"
        )

    # Under scope="crest", attach handlers only to the "crest" logger and
    # isolate it (propagate=False) from whatever sits above it. Under the
    # default scope="root", attach to the root logger instead, so any
    # logger anywhere (not just under "crest") is captured without needing
    # special naming.
    target = logging.getLogger("crest" if scope == "crest" else None)
    _record_original_state(target)
    if scope == "crest":
        target.setLevel(log_level)
    else:
        original_level = _original_logger_state[target][0]
        target.setLevel(min(original_level, log_level))
    if scope == "crest":
        target.propagate = False

    if clear_handlers:
        _remove_handlers(target, owned_only=clear_handlers != "all")

    # Looked up after clearing: clear_handlers="all" also removes loadEnv's
    # handler, and then CREST's own files are the only ones left.
    run_log_handler = (
        _find_run_log_handler() if run_log == "auto" and scope == "root" else None
    )
    use_run_log = run_log_handler is not None

    # File handler for application logs
    if log_file and not use_run_log:
        fh_app = file_handler(log_file)
        fh_app.setLevel(log_level)
        fh_app.setFormatter(formatter)
        _attach_owned(target, fh_app)

    # Error-only file handler
    if error_file and not use_run_log:
        fh_err = file_handler(error_file)
        fh_err.setLevel(logging.ERROR)
        fh_err.setFormatter(formatter)
        _attach_owned(target, fh_err)

    # Console handler
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(console_level)
        ch.setFormatter(formatter)
        _attach_owned(target, ch)

    # Configure metrics logger
    metrics = logging.getLogger("crest.metrics")
    _record_original_state(metrics)
    metrics.setLevel(logging.INFO)
    metrics.propagate = False

    if clear_handlers:
        # Borrowed handlers first, so that "all" can't close the run file.
        _detach_borrowed(metrics)
        _remove_handlers(metrics, owned_only=clear_handlers != "all")

    if not metrics_file:
        # Metrics switched off. Without any handler, logging's last-resort
        # handler would print WARNING+ metrics to stderr.
        _attach_owned(metrics, logging.NullHandler())
    elif use_run_log:
        # Metrics don't propagate, so hand them the run file directly.
        _attach_borrowed(metrics, run_log_handler)
    else:
        fh_met = file_handler(metrics_file)
        fh_met.setLevel(logging.INFO)
        fh_met.setFormatter(formatter)
        _attach_owned(metrics, fh_met)

    # Configure per-module log levels
    if module_levels:
        for module_name, level in module_levels.items():
            module_logger = logging.getLogger(module_name)
            _record_original_state(module_logger)
            parsed_level = _parse_log_level(level)
            module_logger.setLevel(parsed_level)

    # Log the configuration for debugging
    logger = logging.getLogger("crest.utils.crest_logger")
    logger.debug(
        f"Logging configured: scope={scope}, level={logging.getLevelName(log_level)}, "
        f"console={console}, console_level={logging.getLevelName(console_level)}, "
        f"format={format_type}, dir={log_dir}, "
        f"run_log={'used' if use_run_log else 'not used'}"
    )


def _parse_log_level(level: str | int | LogLevel) -> int:
    """
    Parse log level from various input types.

    Parameters
    ----------
    level : str | int | LogLevel
        Log level as string name, integer value, or LogLevel enum.

    Returns
    -------
    int
        Numeric log level.

    Raises
    ------
    ValueError
        If the level string is not a valid log level name.
    """
    if isinstance(level, LogLevel):
        return level.value
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        level_upper = level.upper()
        if hasattr(logging, level_upper):
            return getattr(logging, level_upper)
        raise ValueError(
            f"Invalid log level: {level}. "
            f"Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )
    raise TypeError(
        f"Log level must be str, int, or LogLevel, got {type(level)}")


def install_global_exception_logger(logger_name: str = "crest"):
    """Logs all uncaught exceptions to the given logger."""

    original_hook = sys.excepthook

    def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
        """ Captures and logs uncaught exceptions"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger = logging.getLogger(logger_name)
        logger.critical(
            "Uncaught exception occurred", exc_info=(exc_type, exc_value, exc_traceback)
        )

        original_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = log_uncaught_exceptions
