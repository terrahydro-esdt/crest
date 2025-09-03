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
    def __init__(self, *, fmt_keys: dict[str, str] | None = None):
        super().__init__()
        self.fmt_keys = fmt_keys or {}

    def format(self, record: logging.LogRecord) -> str:
        msg = self._prepare_log_dict(record)
        import json
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

        # Promote selected keys to the front (optional)
        front = {
            key: (always.pop(val, None) if val in always else getattr(record, val, None))
            for key, val in self.fmt_keys.items()
        }
        out = {k: v for k, v in front.items() if v is not None}
        out.update(always)

        # Include extra fields
        for k, v in record.__dict__.items():
            if k not in LOG_RECORD_BUILTIN_ATTRS and k not in out:
                out[k] = v
        return out

def logger_setup(log_file: str = "crest.log.jsonl",
                  metrics_file: str = "metrics.jsonl",
                  console: bool = True) -> None:
    """
    Programmatic logging config:
      - crest -> logs/crest.log.jsonl (+ console)
      - crest.metrics -> logs/metrics.jsonl (no propagation)
    """
    logs_dir = pathlib.Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    app_path = logs_dir / log_file
    met_path = logs_dir / metrics_file

    # Create formatter once
    formatter = CrestJSONFormatter(fmt_keys={
        "level": "levelname",
        "logger": "name",
        "kind": "kind",          # if present in extra
        "stage": "stage",        # if present in extra
        "run": "run",            # if present in extra
    })

    # Root logger (quiet)
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    for h in list(root.handlers):
        root.removeHandler(h)
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    # App logger
    crest = logging.getLogger("crest")
    crest.setLevel(logging.INFO)
    crest.propagate = False
    crest.handlers.clear()
    fh_app = logging.FileHandler(app_path, mode="a", encoding="utf-8")
    fh_app.setLevel(logging.INFO)
    fh_app.setFormatter(formatter)
    crest.addHandler(fh_app)
    if console:
        crest.addHandler(ch)  # share console handler

    # Metrics logger (isolated to its file)
    metrics = logging.getLogger("crest.metrics")
    metrics.setLevel(logging.INFO)
    metrics.propagate = False
    metrics.handlers.clear()
    fh_met = logging.FileHandler(met_path, mode="a", encoding="utf-8")
    fh_met.setLevel(logging.INFO)
    fh_met.setFormatter(formatter)
    metrics.addHandler(fh_met)

def install_global_exception_logger(logger_name: str = "crest"):
    """Logs all uncaught exceptions to the given logger."""

    def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger = logging.getLogger(logger_name)
        logger.critical("Uncaught exception occurred", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = log_uncaught_exceptions
