import datetime as dt
import logging
import logging.config
import pathlib
import sys
import json
import io
import re
import os
import yaml

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
        self.ansi_re = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

    def write(self, buf):
        if not buf:
            return

        buf = self.ansi_re.sub('', buf)
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
    
    def isatty(self):
        """Return False - this is not a real terminal"""
        return False

def logger_setup(config_path: str,
                enabled: bool = True,
                env_vars: dict = None,
                logs_dir = pathlib.Path("logs"),
                redirect_stdio: bool = True):

    try:
        sys.stdout.write('')
        sys.stdout.flush()
        sys.stderr.write('')
        sys.stderr.flush()
    except (BrokenPipeError, OSError, AttributeError):
        try:
            sys.stdout = open('/dev/null', 'w')
            sys.stderr = open('/dev/null', 'w')
        except:
            pass

    if (not enabled):
        logging.disable(logging.CRITICAL)
        print("[logger_setup] Logging disabled")
        return None

    if (not isinstance(logs_dir, pathlib.Path)):
        logs_dir = pathlib.Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # print(f"[logger_setup] Using logging config: {config_path} and logs dir: {logs_dir}")

    with open(config_path) as f:
        raw = f.read()

    if env_vars:
        for key, val in env_vars.items():
            print(f"[logger_setup] Replacing env var {key} with {val}")
            raw = raw.replace("${" + key + "}", val)
    else:
        raw = raw.replace("${LOG_FILE}", str(logs_dir / "crest.log"))

    config = json.loads(raw)

    logging.config.dictConfig(config)

    crest = logging.getLogger("crest")

    if redirect_stdio:
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

def configure_deploy_logging(log_settings_file=None, process_name=None):
    
    logger = None
    try:
        pid = os.getpid()
        date_str = dt.datetime.now().strftime("%Y-%m-%d")

        if (log_settings_file and os.path.exists(log_settings_file)):
            log_settings = {}
            with open(log_settings_file, 'r') as file:
                log_settings = yaml.safe_load(file)

            if process_name:
                log_file = f"{log_settings['logs_dir']}/{process_name}_{date_str}_{pid}.log.jsonl"
            else:
                log_file = f"{log_settings['logs_dir']}/crest_{date_str}_{pid}.log.jsonl"

            logger = logger_setup(
                config_path=log_settings["config_path"],
                logs_dir=log_settings["logs_dir"],
                env_vars={
                    "LOG_FILE": log_file
                },
                redirect_stdio=bool(log_settings["redirect_stdio"])
            )
                
            install_global_exception_logger("crest")
        else:

            default_log_dir = pathlib.Path("/efs/thdro/logs")    
            default_log_dir.mkdir(parents=True, exist_ok=True)

            logs_dir = default_log_dir / f"{date_str}"
            logs_dir.mkdir(parents=True, exist_ok=True)

            logger = logger_setup(config_path="/ASTG/sw/kraken/terrahydro/crest/crest/utils/crest_logging_configs/crest-deploy.json", 
                                  logs_dir=logs_dir,
                                  redirect_stdio=True)
        
            install_global_exception_logger("crest")
    except Exception as e:

        import traceback
        import sys

        if logger is None:
            print(f"Failed to set up logger: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        else:
            logger.exception(f'Failed setting up logger {e}')
        raise e
        
    return logger
