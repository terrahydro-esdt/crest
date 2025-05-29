import json
import logging
import logging.config
import pathlib
import sys
import traceback
from datetime import datetime

from crest import ROOT_PATH


class CrestJSONFormatter(logging.Formatter):
    """Custom JSON log formatter for structured logs with exception trace support."""
    
    def __init__(self, fmt_keys=None):
        super().__init__()
        self.fmt_keys = fmt_keys or {
            "level": "levelname",
            "message": "message",
            "timestamp": "timestamp",
            "logger": "name",
            "module": "module",
            "function": "funcName",
            "line": "lineno",
            "thread_name": "threadName"
        }

    def format(self, record):
        log_record = {}

        for k, v in self.fmt_keys.items():
            if v == "message":
                log_record[k] = record.getMessage()
            else:
                log_record[k] = getattr(record, v, None)

        log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Include exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            log_record["stack"] = self.formatStack(record.stack_info)

        return json.dumps(log_record)


def logger_setup(config_name: str = "default"):
    """Set up the CREST library logger from a JSON config.
    
    Parameters
    ----------
    config_name : str
        Name of the logging config (without .json extension), located in utils/crest_logging_configs/.
    """
    logger = logging.getLogger("crest")

    try:
        config_file = (
            pathlib.Path(ROOT_PATH) /
            "utils" / "crest_logging_configs" / f"{config_name}.json"
        )

        if not config_file.exists():
            raise FileNotFoundError(f"Logging config not found at: {config_file}")

        with open(config_file, "r") as f:
            config = json.load(f)

        # Register custom formatter if used in config
        logging.config.dictConfig(config)

    except Exception as e:
        # Fallback to basic config and log the failure
        logging.basicConfig(level=logging.INFO)
        logger.error("Failed to load structured logging config", exc_info=True)


def install_global_exception_logger(logger_name: str = "crest"):
    """Logs all uncaught exceptions to the given logger."""

    def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger = logging.getLogger(logger_name)
        logger.critical("Uncaught exception occurred", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = log_uncaught_exceptions
