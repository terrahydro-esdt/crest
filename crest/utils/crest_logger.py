import datetime as dt
import logging
import pathlib
import sys
import json
import os
from enum import Enum
from typing import Optional

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


class CrestJSONFormatter(logging.Formatter):
    """ Define a CREST JSON Formatter used by CREST loggers

    Allows for the storage of logs in JSON format so that they can be programmatically
    parsed.

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
        msg = self._prepare_log_dict(record)
        return json.dumps(msg, default=str, ensure_ascii=False)

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict[str, any]:
        always = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.timezone.utc).isoformat(),
        }
        if record.exc_info is not None:
            always["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info is not None:
            always["stack_info"] = self.formatStack(record.stack_info)

        front = {
            key: (always.pop(val, None)
                  if val in always else getattr(record, val, None))
            for key, val in self.fmt_keys.items()
        }
        out = {k: v for k, v in front.items() if v is not None}
        out.update(always)

        for k, v in record.__dict__.items():
            if k not in LOG_RECORD_BUILTIN_ATTRS and k not in out:
                out[k] = v
        return out


def logger_setup(
    log_file: str = "crest.log.jsonl",
    metrics_file: str = "metrics.jsonl",
    console: bool = True,
    log_level: Optional[str | int | LogLevel] = None,
    console_level: Optional[str | int | LogLevel] = None,
    log_dir: str = "logs",
    format_type: str = "json",
    module_levels: Optional[dict[str, str | int]] = None,
    clear_handlers: bool = True
) -> None:
    """
    Configure CREST logging system with enhanced configurability.

    This function sets up the logging infrastructure for CREST, including:
    - Application logs (all log messages)
    - Metrics logs (isolated metrics tracking)
    - Console output (optional)
    - Per-module log level configuration
    - Environment variable support

    Parameters
    ----------
    log_file : str, default="crest.log.jsonl"
        Name of the application log file.
    metrics_file : str, default="metrics.jsonl"
        Name of the metrics log file.
    console : bool, default=True
        Enable console output to stderr.
    log_level : str | int | LogLevel, optional
        Default log level for file output (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        If None, reads from CREST_LOG_LEVEL environment variable, defaults to INFO.
    console_level : str | int | LogLevel, optional
        Log level for console output (can differ from file level).
        If None, reads from CREST_CONSOLE_LOG_LEVEL environment variable,
        defaults to same as log_level.
    log_dir : str, default="logs"
        Directory for log files (relative or absolute path).
    format_type : str, default="json"
        Log format: "json" for structured JSON logs, "text" for human-readable format.
    module_levels : dict[str, str | int], optional
        Per-module log levels for fine-grained control.
        Example: {"crest.model": "DEBUG", "crest.data": "WARNING"}
    clear_handlers : bool, default=True
        Clear existing handlers before adding new ones (prevents duplicate logs).

    Environment Variables
    ---------------------
    CREST_LOG_LEVEL : str
        Default log level if not specified in parameters (e.g., "DEBUG", "INFO")
    CREST_CONSOLE_LOG_LEVEL : str
        Console log level if not specified in parameters
    CREST_LOG_DIR : str
        Log directory if not specified in parameters
    CREST_LOG_FORMAT : str
        Log format ("json" or "text") if not specified in parameters

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

    Using environment variables:

    >>> # export CREST_LOG_LEVEL=DEBUG
    >>> # export CREST_CONSOLE_LOG_LEVEL=WARNING
    >>> logger_setup()  # Will use env vars

    Notes
    -----
    - This function should be called once at application startup
    - Calling it multiple times will reconfigure logging (if clear_handlers=True)
    - The "crest" logger is the root logger for all CREST modules
    - The "crest.metrics" logger is isolated and always logs to metrics_file
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

    # Resolve log directory
    if log_dir == "logs" and "CREST_LOG_DIR" in os.environ:
        log_dir = os.getenv("CREST_LOG_DIR")

    # Resolve format type
    if format_type == "json" and "CREST_LOG_FORMAT" in os.environ:
        format_type = os.getenv("CREST_LOG_FORMAT", "json").lower()

    # Create logs directory
    logs_dir = pathlib.Path(log_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    app_path = logs_dir / log_file
    met_path = logs_dir / metrics_file

    # Create formatter based on format_type
    if format_type == "json":
        formatter = CrestJSONFormatter(fmt_keys={
            "level": "levelname",
            "logger": "name",
            "kind": "kind",
            "stage": "stage",
            "run": "run",
        })
    elif format_type == "text":
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        raise ValueError(f"Invalid format_type: {format_type}. Must be 'json' or 'text'")

    # Configure root CREST logger
    crest = logging.getLogger("crest")
    crest.setLevel(log_level)
    crest.propagate = False

    # Clear existing handlers if requested
    if clear_handlers:
        crest.handlers.clear()

    # File handler for application logs
    fh_app = logging.FileHandler(app_path, mode="a", encoding="utf-8")
    fh_app.setLevel(log_level)
    fh_app.setFormatter(formatter)
    crest.addHandler(fh_app)

    # Console handler (optional)
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(console_level)
        ch.setFormatter(formatter)
        crest.addHandler(ch)

    # Configure metrics logger (isolated to its file)
    metrics = logging.getLogger("crest.metrics")
    metrics.setLevel(logging.INFO)
    metrics.propagate = False

    if clear_handlers:
        metrics.handlers.clear()

    fh_met = logging.FileHandler(met_path, mode="a", encoding="utf-8")
    fh_met.setLevel(logging.INFO)
    fh_met.setFormatter(formatter)
    metrics.addHandler(fh_met)

    # Configure per-module log levels
    if module_levels:
        for module_name, level in module_levels.items():
            module_logger = logging.getLogger(module_name)
            parsed_level = _parse_log_level(level)
            module_logger.setLevel(parsed_level)

    # Log the configuration for debugging
    logger = logging.getLogger("crest.utils.crest_logger")
    logger.debug(
        f"Logging configured: level={logging.getLevelName(log_level)}, "
        f"console={console}, console_level={logging.getLevelName(console_level)}, "
        f"format={format_type}, dir={log_dir}"
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
    raise TypeError(f"Log level must be str, int, or LogLevel, got {type(level)}")


def install_global_exception_logger(logger_name: str = "crest"):
    """Logs all uncaught exceptions to the given logger."""

    original_hook = sys.excepthook

    def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger = logging.getLogger(logger_name)
        logger.critical("Uncaught exception occurred",
                        exc_info=(exc_type, exc_value, exc_traceback))
        
        original_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = log_uncaught_exceptions
