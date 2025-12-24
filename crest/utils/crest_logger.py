import datetime as dt
import logging
import pathlib
import sys
import json
import io
import re

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

class StreamToLogger(io.TextIOBase):

    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, buf):
        ansi_re = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

        if not buf:
            return
        # Strip ANSI sequences
        buf = ansi_re.sub('', buf)
        self._buffer += buf
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        if self._buffer:
            line = self._buffer.strip()
            if line:
                self.logger.log(self.level, line)
            self._buffer = ""

def logger_setup(config_path: str,
                enabled: bool = True,
                env_vars: dict = None,
                logs_dir = pathlib.Path("logs")):

    if (not enabled):
        logging.disable(logging.CRITICAL)
        print("[logger_setup] Logging disabled")
        return 

    if (not isinstance(logs_dir, pathlib.Path)):
        logs_dir = pathlib.Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[logger_setup] Using logging config: {config_path} and logs dir: {logs_dir}")

    with open(config_path) as f:
        raw = f.read()

    if env_vars:
        for key, val in env_vars.items():
            print(f"[logger_setup] Replacing env var {key} with {val}")
            raw = raw.replace("${" + key + "}", val)

    config = json.loads(raw)
    logging.config.dictConfig(config)

    crest = logging.getLogger("crest")
    sys.stdout = StreamToLogger(crest, logging.INFO)
    sys.stderr = StreamToLogger(crest, logging.ERROR)

    crest.info("Crest logger initialized.")
    for h in crest.handlers:
        crest.info(f"Crest logger handler: {h}")
        if isinstance(h, logging.FileHandler):
            crest.info(f"  -> log file: {h.baseFilename}")

    return crest

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
