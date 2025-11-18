# CREST logging

### LoggingService

```python
from crest.utils.logging_service import LoggingService, get_logger

# Initialize logging (call once at application startup)
LoggingService.initialize()

# Get a logger for your module
logger = get_logger(__name__)

# Use the logger
logger.info("application started")
logger.debug("detailed debug information")
logger.warning("warning")
logger.error("error occurred")
```

### Backward compatibility

```python
from crest.utils.crest_logger import logger_setup
import logging

# Initialize logging
logger_setup()

# Get a logger
logger = logging.getLogger(__name__)
logger.info("Application started")
```

**Note:** `LoggingService.initialize()` calls `logger_setup()` internally, so they provide the same core functionality.

## Examples

### Development: use verbose logging

```python
from crest.utils.logging_service import LoggingService

# Enable DEBUG level for development
LoggingService.initialize(log_level="DEBUG", format_type="text")
```

### Production: use "quiet console"

```python
from crest.utils.logging_service import LoggingService

# Log everything to file, but only warnings/errors to console
LoggingService.initialize(
    log_level="INFO",
    console_level="WARNING"
)
```

### Log Levels

Use string names, integers, or the `LogLevel` enum:

```python
from crest.utils.logging_service import LoggingService
from crest.utils.crest_logger import LogLevel

# Set either way:
LoggingService.initialize(log_level="DEBUG")
LoggingService.initialize(log_level=10)  # logging.DEBUG
LoggingService.initialize(log_level=LogLevel.DEBUG)
```

Available levels (from most to least verbose):
- `DEBUG` (10): Detailed diagnostic information
- `INFO` (20): General informational messages
- `WARNING` (30): Warning messages
- `ERROR` (40): Error messages
- `CRITICAL` (50): Critical errors

## Environment Variables

Configure logging via environment variables:

```bash
# Set default log level
export CREST_LOG_LEVEL=DEBUG

# Set console log level (can differ from file)
export CREST_CONSOLE_LOG_LEVEL=WARNING

# Set log directory
export CREST_LOG_DIR=/s3/thdro/crest

# Set format
export CREST_LOG_FORMAT=text
```

Then:
```python
from crest.utils.logging_service import LoggingService

LoggingService.initialize()  # Will use environment variables
```

Or with the alternative API:
```python
from crest.utils.crest_logger import logger_setup

logger_setup()  # Will also use environment variables
```

## Enhancements

### Log rotation

Prevent log files from growing unbounded with automatic rotation:

#### Size-based rotation

```python
from crest.utils.logging_service import LoggingService

# Rotate when log reaches 10MB, keep 5 backups
LoggingService.initialize(
    rotate=True,
    max_bytes=10 * 1024 * 1024,  # 10MB
    backup_count=5
)
```

#### Time-based rotation

```python
from crest.utils.logging_service import LoggingService

# Rotate daily at midnight, keep 30 days
LoggingService.initialize(
    rotate=True,
    rotate_when='midnight',
    backup_count=30
)

# Rotate every hour, keep 24 hours
LoggingService.initialize(
    rotate=True,
    rotate_when='H',
    backup_count=24
)

# Rotate every Monday, keep 4 weeks
LoggingService.initialize(
    rotate=True,
    rotate_when='W0',  # W0=Monday, W6=Sunday
    backup_count=4
)
```

**Time-based rotation options:**

- `'S'` - Seconds
- `'M'` - Minutes
- `'H'` - Hours
- `'D'` - Days
- `'midnight'` - Daily at midnight
- `'W0'` through `'W6'` - Weekly (Monday=W0, Sunday=W6)

### Per-module configuration

Control verbosity for specific modules:

```python
from crest.utils.logging_service import LoggingService

LoggingService.initialize(
    log_level="INFO",  # Default for all modules
    module_levels={
        "crest.model": "DEBUG",           # Verbose model logs
        "crest.data.batching": "DEBUG",   # Verbose batching logs
        "crest.archiver": "WARNING",      # Quiet archiver logs
        "crest.data_server": "ERROR"      # Very quiet data server
    }
)
```

## Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_file` | str | `"crest.log.jsonl"` | Application log filename |
| `metrics_file` | str | `"metrics.jsonl"` | Metrics log filename |
| `console` | bool | `True` | Enable console output |
| `log_dir` | str | `"logs"` | Directory for log files |
| `format_type` | str | `"json"` | Format: "json" or "text" |
| `module_levels` | dict | `None` | Per-module log levels |
| `clear_handlers` | bool | `True` | Clear existing handlers |
| `rotate` | bool | `False` | Enable log rotation |
| `max_bytes` | int | `10485760` (10MB) | Max log size before rotation |
| `backup_count` | int | `5` | Number of backup files to keep |
| `rotate_when` | str | `None` | Time-based rotation interval |


## References

- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
