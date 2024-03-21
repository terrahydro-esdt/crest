import datetime as dt
import json
import logging

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
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

#    @override
    def format(self, record: logging.LogRecord) -> str:
        """ Takes a log record object and convert to a string """
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

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
