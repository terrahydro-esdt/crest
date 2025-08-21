import datetime as dt
import json
import logging
import pathlib,sys
import os
import logging.config

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

ROOT_PATH = pathlib.Path(__file__).parent.parent.resolve()

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
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    def format(self, record: logging.LogRecord) -> str:
        """ Takes a log record object and convert to a string """
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str, ensure_ascii=False)

    def _prepare_log_dict(self, record: logging.LogRecord):
        """ Helper function to format the log record"""
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.timezone.utc).isoformat(),
        }
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: msg_val
            if (msg_val := always_fields.pop(val, None)) is not None
            else getattr(record, val)
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val
        return message

def logger_setup(config_name: str = "crest-logfile", log_file = None):
    """Set up the CREST library logger from a JSON config.
    
    Parameters
    ----------
    config_name : str
        Name of the logging config (without .json extension), located in utils/crest_logging_configs/.
    """

    logger = logging.getLogger("crest")

    try:
        config_file = (
            ROOT_PATH /
            "utils" / "crest_logging_configs" / f"{config_name}.json"
        )

        if not config_file.exists():
            raise FileNotFoundError(f"Logging config not found at: {config_file}")

        config_raw = {}
        with open(config_file, "r") as f:
            config_raw = json.load(f)

        log_file = log_file if log_file is not None else os.environ.get("LOG_FILE", "logs/crest.log.jsonl")

        # Ensure the parent directory exists
        log_dir = pathlib.Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Now replace placeholder in config
        config_raw_str = json.dumps(config_raw)
        config_raw_str = config_raw_str.replace("${LOG_FILE}", log_file)
        config = json.loads(config_raw_str)

        logging.config.dictConfig(config)


    except Exception as e:
        print(f"Failed to load structured logging config: {e}")
        fallback_log_file = "logs/crest_fallback.log"
        pathlib.Path(fallback_log_file).parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(fallback_log_file), logging.StreamHandler()]
        )
        logging.getLogger().error("Logging fallback activated", exc_info=True)



def install_global_exception_logger(logger_name: str = "crest"):
    """Logs all uncaught exceptions to the given logger."""

    def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger = logging.getLogger(logger_name)
        logger.critical("Uncaught exception occurred", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = log_uncaught_exceptions
