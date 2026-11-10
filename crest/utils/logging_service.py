"""
This module provides a unified interface for configuring and accessing loggers
throughout the CREST application. It wraps logger_setup functionality.

By default (`scope="root"`), initializing logging captures any logger in the
process obtained via the standard `logging.getLogger(__name__)` pattern - not
just loggers named "crest.*" - so applications embedding CREST do not need to
follow any CREST-specific naming convention. Pass `scope="crest"` to isolate
CREST's own logs instead; see `crest.utils.crest_logger.logger_setup` for
details.

Examples
--------
Initialize logging at application startup:

>>> from crest.utils.logging_service import LoggingService
>>> LoggingService.initialize(log_level="INFO", console_level="WARNING")

Then get a logger in any module:

>>> from crest.utils.logging_service import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Application started")

Can also use context-aware logging:

>>> from crest.utils.logging_service import log_context
>>> with log_context(run="exp_001", stage="training"):
...     logger.info("Training started")  # Will include run and stage
"""
from __future__ import annotations

import inspect
import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Literal, Optional

from .crest_logger import LogLevel, _reset_logging_state, logger_setup


class LoggingService:
    """
    Centralized logging service for CREST.

    This class provides a singleton-like interface for managing logging
    configuration across the entire application. It ensures logging is
    initialized once and provides convenient methods for accessing loggers.

    Attributes
    ----------
    _initialized : bool
        Whether the logging system has been initialized
    _config : dict
        Current logging configuration parameters

    Examples
    --------
    Initialize logging at application startup:

    >>> LoggingService.initialize(log_level="DEBUG")

    Get a logger:

    >>> logger = LoggingService.get_logger("crest.model")
    >>> logger.info("Model loaded")

    Reconfigure logging at runtime:

    >>> LoggingService.reconfigure(log_level="WARNING")
    """

    _initialized: bool = False
    _config: dict = {}

    @classmethod
    def initialize(
        cls,
        log_file: Optional[str] = "crest.log.jsonl",
        metrics_file: Optional[str] = "metrics.jsonl",
        error_file: Optional[str] = "error.jsonl",
        console: bool = True,
        log_level: Optional[str | int | LogLevel] = None,
        console_level: Optional[str | int | LogLevel] = None,
        scope: Optional[str] = None,
        log_dir: str = "logs",
        format_type: str = "json",
        module_levels: Optional[dict[str, str | int]] = None,
        clear_handlers: bool | Literal["all"] = True,
        rotate: bool = False,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        rotate_when: Optional[str] = None,
        run_log: Literal["auto", "ignore"] = "auto",
        force: bool = False,
    ) -> None:
        """
        Initialize the CREST logging system.

        This method should be called once at application startup. Subsequent
        calls will be ignored unless force=True.

        Parameters
        ----------
        log_file : str or None, default="crest.log.jsonl"
            Name of the application log file; None writes none. The three
            file names may contain ``{pid}`` and ``{host}`` placeholders (see
            `crest_logger.logger_setup`)
        metrics_file : str or None, default="metrics.jsonl"
            Name of the metrics log file; None drops metrics records
        error_file : str or None, default="error.jsonl"
            Name of the errors-only log file; None writes none
        console : bool, default=True
            Enable console output
        log_level : str | int | LogLevel, optional
            Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_level : str | int | LogLevel, optional
            Console log level (can differ from file level)
        scope : "root" | "crest", optional
            Where handlers attach - "root" (default) captures any caller's
            `logging.getLogger(__name__)`; "crest" isolates CREST's logs
            under the "crest" logger only. See `crest_logger.logger_setup`
            for full details.
        log_dir : str, default="logs"
            Directory for log files
        format_type : str, default="json"
            Format: "json" or "text"
        module_levels : dict[str, str | int], optional
            Per-module log levels
        clear_handlers : bool | "all", default=True
            Which existing handlers to remove before (re)initializing: True
            removes only handlers CREST installed earlier, leaving the host's,
            pytest's or other libraries' handlers in place; False removes
            nothing; "all" removes every handler on the target logger. See
            `crest_logger.logger_setup` for details.
        rotate : bool, default=False
            Enable log file rotation to prevent unbounded file growth
        max_bytes : int, default=10485760 (10MB)
            Maximum size of log file before rotation (size-based rotation,
            used when rotate=True and rotate_when=None)
        backup_count : int, default=5
            Number of backup log files to keep when rotating
        rotate_when : str, optional
            Time-based rotation interval ('S', 'M', 'H', 'D', 'midnight',
            'W0'-'W6'); if specified, uses time-based rotation instead of
            size-based
        run_log : "auto" | "ignore", default="auto"
            With "auto", when loadEnv's run log is on the root logger, CREST
            doesn't write its own application and error log files, since
            records already reach the run file, and sends metrics to the run
            file instead of `metrics_file`. "ignore" writes CREST's files anyway.
            See `crest_logger.logger_setup` for details.
        force : bool, default=False
            Force reinitialization even if already initialized

        Examples
        --------
        Basic initialization:

        >>> LoggingService.initialize()

        Custom configuration:

        >>> LoggingService.initialize(
        ...     log_level="DEBUG",
        ...     console_level="WARNING",
        ...     format_type="text",
        ...     module_levels={"crest.model": "DEBUG"}
        ... )
        """
        if cls._initialized and not force:
            logger = logging.getLogger("crest.utils.logging_service")
            logger.debug("Logging already initialized (use force=True to reinitialize)")
            return

        # Store configuration
        cls._config = {
            "log_file": log_file,
            "metrics_file": metrics_file,
            "error_file": error_file,
            "console": console,
            "log_level": log_level,
            "console_level": console_level,
            "scope": scope,
            "log_dir": log_dir,
            "format_type": format_type,
            "module_levels": module_levels,
            "clear_handlers": clear_handlers,
            "rotate": rotate,
            "max_bytes": max_bytes,
            "backup_count": backup_count,
            "rotate_when": rotate_when,
            "run_log": run_log,
        }

        # Initialize logging
        logger_setup(
            log_file=log_file,
            metrics_file=metrics_file,
            error_file=error_file,
            console=console,
            log_level=log_level,
            console_level=console_level,
            scope=scope,
            log_dir=log_dir,
            format_type=format_type,
            module_levels=module_levels,
            clear_handlers=clear_handlers,
            rotate=rotate,
            max_bytes=max_bytes,
            backup_count=backup_count,
            rotate_when=rotate_when,
            run_log=run_log,
        )

        cls._initialized = True

        logger = logging.getLogger("crest.utils.logging_service")
        logger.debug("LoggingService initialized successfully")

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance.

        If logging hasn't been initialized, this will auto-initialize with
        default settings.

        Parameters
        ----------
        name : str
            Logger name (typically __name__)

        Returns
        -------
        logging.Logger
            A logger with exactly the given `name` - this never rewrites
            or prefixes `name` with "crest." Under the default
            `scope="root"`, any logger name is captured by CREST's
            configured handlers regardless.

        Examples
        --------
        >>> logger = LoggingService.get_logger(__name__)
        >>> logger.info("Processing started")
        """
        if not cls._initialized:
            # Auto-initialize with defaults
            cls.initialize()

        return logging.getLogger(name)

    @classmethod
    def reconfigure(cls, **kwargs) -> None:
        """
        Reconfigure logging at runtime.

        This method allows changing logging configuration after initialization.
        All parameters from initialize() are accepted.

        Parameters
        ----------
        **kwargs
            Any parameters accepted by initialize()

        Examples
        --------
        Change log level:

        >>> LoggingService.reconfigure(log_level="DEBUG")

        Enable text format:

        >>> LoggingService.reconfigure(format_type="text")
        """
        # Merge with existing config
        config = cls._config.copy()
        config.update(kwargs)

        # Reinitialize
        cls._initialized = False
        cls.initialize(**config)

    @classmethod
    def is_initialized(cls) -> bool:
        """
        Check if logging has been initialized.

        Returns
        -------
        bool
            True if initialized, False otherwise
        """
        return cls._initialized

    @classmethod
    def get_config(cls) -> dict:
        """
        Get current logging configuration.

        Returns
        -------
        dict
            Current configuration parameters
        """
        return cls._config.copy()

    @classmethod
    def reset(cls) -> None:
        """
        Reset the logging service to uninitialized state.

        Detaches the handlers CREST borrowed (loadEnv's run-file handler on
        "crest.metrics") without closing them, removes (and closes) the
        handlers CREST installed, and restores the
        levels and propagate flags CREST changed - on the root logger,
        "crest", "crest.metrics" and any `module_levels` loggers. Handlers
        and settings CREST did not install or change (e.g. pytest's log
        capture, or a host application's own root handlers) are left alone.

        This is primarily useful for testing.
        """
        cls._initialized = False
        cls._config = {}
        _reset_logging_state()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Convenience function to get a logger instance.

    If name is None, automatically detects the caller's module name.

    Parameters
    ----------
    name : str, optional
        Logger name. If None, uses caller's __name__

    Returns
    -------
    logging.Logger
        A logger with exactly `name` (or the detected caller's `__name__`) -
        never rewritten or prefixed with "crest." Under the default
        `scope="root"` this is captured by CREST's configured handlers
        regardless of the name's package.

    Examples
    --------
    Explicit name:

    >>> logger = get_logger(__name__)

    Auto-detect name:

    >>> logger = get_logger()  # Uses caller's __name__
    """
    if name is None:
        # Auto-detect caller's module name
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_globals = frame.f_back.f_globals
            name = caller_globals.get("__name__", "crest")

    return LoggingService.get_logger(name)


# Fields set by log_context(). A ContextVar, so each thread and each asyncio
# task has its own; the dict is replaced, never mutated.
_log_context: ContextVar[dict[str, Any]] = ContextVar("crest_log_context", default={})

# Marks CREST's makeRecord wrapper, so it's installed only once.
_CONTEXT_WRAPPER_ATTR = "_crest_log_context"
_install_lock = threading.Lock()


def _install_context_stamping() -> None:
    """Wrap ``logging.Logger.makeRecord`` (once) to add the current log_context fields.

    The fields are set after ``makeRecord`` has applied the call's ``extra``,
    and only for keys that aren't in it, so ``extra`` wins and passing the same
    key never raises. Whatever ``makeRecord`` was already (e.g. loadEnv's
    stamping patch) is called first, so context fields also win over fields it
    stamps from the environment. In the other order (a patch installed on top
    of this one that fills only missing fields), the result is the same.
    """
    with _install_lock:
        original = logging.Logger.makeRecord
        if getattr(original, _CONTEXT_WRAPPER_ATTR, False):
            return

        @wraps(original)
        def make_record(self, name, level, fn, lno, msg, args, exc_info,
                        func=None, extra=None, sinfo=None):
            record = original(self, name, level, fn, lno, msg, args, exc_info,
                              func, extra, sinfo)
            for key, value in _log_context.get().items():
                if extra is None or key not in extra:
                    setattr(record, key, value)
            return record

        setattr(make_record, _CONTEXT_WRAPPER_ATTR, True)
        logging.Logger.makeRecord = make_record


@contextmanager
def log_context(**kwargs):
    """
    Context manager to add fields to all log messages within the block.

    This is useful for adding structured logging context like a run or
    experiment label, stage, etc. to all log messages within a specific code
    block. Nested contexts merge, and each thread and asyncio task has its own
    context: records logged by another thread while the block runs don't get
    these fields. A field passed explicitly with ``extra=`` in a logging call
    takes precedence over the context's value.

    Use ``run`` for a run or experiment label, not ``run_id``: under loadEnv,
    ``run_id`` is the process's run ID, and replacing it would take the records
    out of that run in ``runlog``.

    Parameters
    ----------
    **kwargs
        Key-value pairs to add to log records

    Yields
    ------
    None

    Examples
    --------
    Add run and stage to all logs:

    >>> logger = get_logger(__name__)
    >>> with log_context(run="exp_001", stage="training"):
    ...     logger.info("Starting epoch 1")
    ...     # Log will include: run="exp_001", stage="training"

    Nested contexts:

    >>> with log_context(run="exp_001"):
    ...     with log_context(stage="training"):
    ...         logger.info("Training started")
    ...         # Includes both run and stage
    ...     with log_context(stage="evaluation"):
    ...         logger.info("Evaluation started")
    ...         # run from outer context, stage="evaluation"

    A per-call field wins over the context:

    >>> with log_context(stage="training"):
    ...     logger.info("Fetching weights", extra={"stage": "download"})
    ...     # stage="download"
    """
    _install_context_stamping()
    token = _log_context.set({**_log_context.get(), **kwargs})
    try:
        yield
    finally:
        _log_context.reset(token)


def log_function_call(logger: Optional[logging.Logger] = None, level: str = "DEBUG"):
    """
    Decorator to log function calls with arguments and return values.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use. If None, creates one from function's module
    level : str, default="DEBUG"
        Log level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns
    -------
    callable
        Decorated function

    Examples
    --------
    Log all calls to a function:

    >>> @log_function_call()
    ... def process_data(x, y):
    ...     return x + y

    Use specific logger and level:

    >>> logger = get_logger(__name__)
    >>> @log_function_call(logger=logger, level="INFO")
    ... def important_function():
    ...     pass
    """

    def decorator(func):
        """ Wraps `func` so each call is logged with its arguments and return value """
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        log_method = getattr(logger, level.lower())

        @wraps(func)
        def wrapper(*args, **kwargs):
            """ Logs the call signature and return value (or exception) around a call to `func` """
            # Log function entry
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            log_method(f"Calling {func.__name__}({signature})")

            try:
                result = func(*args, **kwargs)
                log_method(f"{func.__name__} returned {result!r}")
                return result
            except Exception as e:
                logger.exception(f"{func.__name__} raised {type(e).__name__}")
                raise

        return wrapper

    return decorator


def log_execution_time(logger: Optional[logging.Logger] = None, level: str = "INFO"):
    """
    Decorator to log function execution time.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use. If None, creates one from function's module
    level : str, default="INFO"
        Log level to use

    Returns
    -------
    callable
        Decorated function

    Examples
    --------
    >>> @log_execution_time()
    ... def slow_function():
    ...     time.sleep(1)
    ...     return "done"
    """
    import time

    def decorator(func):
        """ Wraps `func` so each call is timed and the elapsed duration is logged """
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        log_method = getattr(logger, level.lower())

        @wraps(func)
        def wrapper(*args, **kwargs):
            """ Times a call to `func` and logs the elapsed duration (or failure) """
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                log_method(
                    f"{func.__name__} completed in {elapsed:.3f}s",
                    extra={"duration_sec": elapsed, "function": func.__name__},
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.exception(
                    f"{func.__name__} failed after {elapsed:.3f}s",
                    extra={"duration_sec": elapsed, "function": func.__name__},
                )
                raise

        return wrapper

    return decorator


# Convenience exports
__all__ = [
    "LoggingService",
    "get_logger",
    "log_context",
    "log_function_call",
    "log_execution_time",
]
