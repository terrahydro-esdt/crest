"""
This module provides a unified interface for configuring and accessing loggers
throughout the CREST application. It wraps logger_setup functionality.

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
>>> with log_context(run_id="exp_001", stage="training"):
...     logger.info("Training started")  # Will include run_id and stage
"""

import inspect
import logging
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Optional

from .crest_logger import LogLevel, logger_setup


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
        log_file: str = "crest.log.jsonl",
        metrics_file: str = "metrics.jsonl",
        console: bool = True,
        log_level: Optional[str | int | LogLevel] = None,
        console_level: Optional[str | int | LogLevel] = None,
        log_dir: str = "logs",
        format_type: str = "json",
        module_levels: Optional[dict[str, str | int]] = None,
        clear_handlers: bool = True,
        force: bool = False,
    ) -> None:
        """
        Initialize the CREST logging system.

        This method should be called once at application startup. Subsequent
        calls will be ignored unless force=True.

        Parameters
        ----------
        log_file : str, default="crest.log.jsonl"
            Name of the application log file
        metrics_file : str, default="metrics.jsonl"
            Name of the metrics log file
        console : bool, default=True
            Enable console output
        log_level : str | int | LogLevel, optional
            Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_level : str | int | LogLevel, optional
            Console log level (can differ from file level)
        log_dir : str, default="logs"
            Directory for log files
        format_type : str, default="json"
            Format: "json" or "text"
        module_levels : dict[str, str | int], optional
            Per-module log levels
        clear_handlers : bool, default=True
            Clear existing handlers
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
            "console": console,
            "log_level": log_level,
            "console_level": console_level,
            "log_dir": log_dir,
            "format_type": format_type,
            "module_levels": module_levels,
            "clear_handlers": clear_handlers,
        }

        # Initialize logging
        logger_setup(
            log_file=log_file,
            metrics_file=metrics_file,
            console=console,
            log_level=log_level,
            console_level=console_level,
            log_dir=log_dir,
            format_type=format_type,
            module_levels=module_levels,
            clear_handlers=clear_handlers,
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
            Configured logger instance

        Examples
        --------
        >>> logger = LoggingService.get_logger(__name__)
        >>> logger.info("Processing started")
        """
        if not cls._initialized:
            # Auto-initialize with defaults
            cls.initialize()

        # Ensure logger is under crest hierarchy
        if not name.startswith("crest"):
            name = f"crest.{name}"

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

        This is primarily useful for testing.
        """
        cls._initialized = False
        cls._config = {}

        # Clear all handlers from crest loggers
        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            if logger_name.startswith("crest"):
                logger = logging.getLogger(logger_name)
                logger.handlers.clear()
                logger.setLevel(logging.NOTSET)


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
        Configured logger instance

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


@contextmanager
def log_context(**kwargs):
    """
    Context manager to add fields to all log messages within the block.

    This is useful for adding structured logging context like run_id, stage,
    experiment name, etc. to all log messages within a specific code block.

    Parameters
    ----------
    **kwargs
        Key-value pairs to add to log records

    Yields
    ------
    None

    Examples
    --------
    Add run_id and stage to all logs:

    >>> with log_context(run_id="exp_001", stage="training"):
    ...     logger.info("Starting epoch 1")
    ...     # Log will include: run_id="exp_001", stage="training"

    Nested contexts:

    >>> with log_context(run_id="exp_001"):
    ...     with log_context(stage="training"):
    ...         logger.info("Training started")
    ...         # Includes both run_id and stage
    ...     with log_context(stage="evaluation"):
    ...         logger.info("Evaluation started")
    ...         # run_id from outer context, stage="evaluation"
    """
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kw):
        record = old_factory(*args, **kw)
        for key, value in kwargs.items():
            setattr(record, key, value)
        return record

    logging.setLogRecordFactory(record_factory)
    try:
        yield
    finally:
        logging.setLogRecordFactory(old_factory)


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
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        log_method = getattr(logger, level.lower())

        @wraps(func)
        def wrapper(*args, **kwargs):
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
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        log_method = getattr(logger, level.lower())

        @wraps(func)
        def wrapper(*args, **kwargs):
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
