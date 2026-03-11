import pytest
import logging
import os
import json
import tempfile
from pathlib import Path

from crest.utils.crest_logger import (
    logger_setup,
    LogLevel,
    _parse_log_level,
    CrestJSONFormatter
)


@pytest.fixture
def temp_log_dir(tmp_path):
    """Temporary directory for log files."""
    return str(tmp_path)


@pytest.fixture(autouse=True)
def cleanup_loggers():
    """Clean up after each test."""
    yield
    # Clear all handlers from crest loggers
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if logger_name.startswith('crest'):
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)


@pytest.fixture(autouse=True)
def cleanup_env_vars():
    """Clean up after each test."""
    env_vars = [
        'CREST_LOG_LEVEL',
        'CREST_CONSOLE_LOG_LEVEL',
        'CREST_LOG_DIR',
        'CREST_LOG_FORMAT'
    ]
    original_values = {var: os.environ.get(var) for var in env_vars}

    yield

    # Restore original environment
    for var, value in original_values.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


class TestParseLogLevel:

    """ See https://docs.python.org/3.12/library/logging.html#logging-levels """

    def test_parse_string_levels(self):
        assert _parse_log_level('DEBUG') == logging.DEBUG
        assert _parse_log_level('INFO') == logging.INFO
        assert _parse_log_level('WARNING') == logging.WARNING
        assert _parse_log_level('ERROR') == logging.ERROR
        assert _parse_log_level('CRITICAL') == logging.CRITICAL

    def test_parse_lowercase_string(self):
        assert _parse_log_level('debug') == logging.DEBUG
        assert _parse_log_level('info') == logging.INFO

    def test_parse_int_levels(self):
        assert _parse_log_level(10) == logging.DEBUG
        assert _parse_log_level(20) == logging.INFO
        assert _parse_log_level(30) == logging.WARNING

    def test_parse_enum_levels(self):
        assert _parse_log_level(LogLevel.DEBUG) == logging.DEBUG
        assert _parse_log_level(LogLevel.INFO) == logging.INFO
        assert _parse_log_level(LogLevel.WARNING) == logging.WARNING

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Invalid log level"):
            _parse_log_level('INVALID')

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Log level must be"):
            _parse_log_level([])


class TestLoggerSetupBasic:

    def test_basic_setup(self, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, console=False)

        # Check that crest logger exists and is configured
        crest_logger = logging.getLogger('crest')
        assert crest_logger.level == logging.INFO
        assert len(crest_logger.handlers) > 0

        # Check that log file was created
        log_file = Path(temp_log_dir) / 'crest.log.jsonl'
        assert log_file.exists()

    def test_custom_log_files(self, temp_log_dir):
        logger_setup(
            log_file='custom_app.log',
            metrics_file='custom_metrics.log',
            log_dir=temp_log_dir,
            console=False
        )

        assert (Path(temp_log_dir) / 'custom_app.log').exists()
        assert (Path(temp_log_dir) / 'custom_metrics.log').exists()

    def test_log_level_debug(self, temp_log_dir):
        logger_setup(log_level='DEBUG', log_dir=temp_log_dir, console=False)

        crest_logger = logging.getLogger('crest')
        assert crest_logger.level == logging.DEBUG

    def test_log_level_warning(self, temp_log_dir):
        logger_setup(log_level=LogLevel.WARNING,
                     log_dir=temp_log_dir, console=False)

        crest_logger = logging.getLogger('crest')
        assert crest_logger.level == logging.WARNING

    def test_console_disabled(self, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, console=False)

        crest_logger = logging.getLogger('crest')
        # Should only have file handler, not stream handler
        stream_handlers = [h for h in crest_logger.handlers
                           if isinstance(h, logging.StreamHandler)
                           and not isinstance(h, logging.FileHandler)]
        assert len(stream_handlers) == 0


class TestLoggerSetupFormats:
    def test_json_format(self, temp_log_dir):
        logger_setup(
            format_type='json',
            log_dir=temp_log_dir,
            console=False
        )

        logger = logging.getLogger('crest.test')
        logger.info('Test message')

        # Read the log file and verify JSON format
        log_file = Path(temp_log_dir) / 'crest.log.jsonl'
        with open(log_file) as f:
            line = f.readline()
            log_entry = json.loads(line)
            assert 'message' in log_entry
            assert 'timestamp' in log_entry
            assert 'level' in log_entry
            assert log_entry['message'] == 'Test message'

    def test_text_format(self, temp_log_dir):
        logger_setup(
            format_type='text',
            log_dir=temp_log_dir,
            console=False
        )

        logger = logging.getLogger('crest.test')
        logger.info('Test message')

        # Read the log file and verify text format
        log_file = Path(temp_log_dir) / 'crest.log.jsonl'
        with open(log_file) as f:
            line = f.readline()
            # Should not be JSON
            with pytest.raises(json.JSONDecodeError):
                json.loads(line)
            # Should contain expected text elements
            assert 'INFO' in line
            assert 'crest.test' in line
            assert 'Test message' in line

    def test_invalid_format_raises(self, temp_log_dir):
        with pytest.raises(ValueError, match="Invalid format_type"):
            logger_setup(format_type='invalid', log_dir=temp_log_dir)


class TestModuleLevels:

    def test_module_levels(self, temp_log_dir):
        logger_setup(
            log_level='INFO',
            module_levels={
                'crest.test.verbose': 'DEBUG',
                'crest.test.quiet': 'ERROR'
            },
            log_dir=temp_log_dir,
            console=False
        )

        verbose_logger = logging.getLogger('crest.test.verbose')
        quiet_logger = logging.getLogger('crest.test.quiet')
        normal_logger = logging.getLogger('crest.test.normal')

        assert verbose_logger.level == logging.DEBUG
        assert quiet_logger.level == logging.ERROR
        assert normal_logger.level == logging.NOTSET  # Inherits from parent

    def test_module_levels_with_int(self, temp_log_dir):
        logger_setup(
            log_level='INFO',
            module_levels={
                'crest.test.custom': 15  # Between DEBUG and INFO
            },
            log_dir=temp_log_dir,
            console=False
        )

        custom_logger = logging.getLogger('crest.test.custom')
        assert custom_logger.level == 15


class TestEnvironmentVariables:

    def test_env_log_level(self, temp_log_dir):
        os.environ['CREST_LOG_LEVEL'] = 'DEBUG'

        logger_setup(log_dir=temp_log_dir, console=False)

        crest_logger = logging.getLogger('crest')
        assert crest_logger.level == logging.DEBUG

    def test_env_console_level(self, temp_log_dir):
        os.environ['CREST_LOG_LEVEL'] = 'DEBUG'
        os.environ['CREST_CONSOLE_LOG_LEVEL'] = 'ERROR'

        logger_setup(log_dir=temp_log_dir, console=True)

        crest_logger = logging.getLogger('crest')

        # Find the console handler
        stream_handler = None
        for handler in crest_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                stream_handler = handler
                break

        assert stream_handler is not None
        assert stream_handler.level == logging.ERROR

    def test_env_log_format(self, temp_log_dir):
        os.environ['CREST_LOG_FORMAT'] = 'text'

        logger_setup(log_dir=temp_log_dir, console=False)

        logger = logging.getLogger('crest.test')
        logger.info('Test')

        # Verify text format was used
        log_file = Path(temp_log_dir) / 'crest.log.jsonl'
        with open(log_file) as f:
            line = f.readline()
            with pytest.raises(json.JSONDecodeError):
                json.loads(line)

    def test_parameter_overrides_env(self, temp_log_dir):
        os.environ['CREST_LOG_LEVEL'] = 'DEBUG'

        logger_setup(log_level='WARNING', log_dir=temp_log_dir, console=False)

        crest_logger = logging.getLogger('crest')
        assert crest_logger.level == logging.WARNING


class TestStructuredLogging:

    def test_extra_fields_in_json(self, temp_log_dir):
        logger_setup(
            format_type='json',
            log_dir=temp_log_dir,
            console=False
        )

        logger = logging.getLogger('crest.test')
        logger.info(
            'Batch processed',
            extra={
                'batch_size': 32,
                'duration_sec': 1.234,
                'stage': 'training',
                'run': 'exp_001'
            }
        )

        # Read and verify extra fields are present
        log_file = Path(temp_log_dir) / 'crest.log.jsonl'
        with open(log_file) as f:
            line = f.readline()
            log_entry = json.loads(line)
            assert log_entry['batch_size'] == 32
            assert log_entry['duration_sec'] == 1.234
            assert log_entry['stage'] == 'training'
            assert log_entry['run'] == 'exp_001'


class TestMetricsLogger:

    def test_metrics_logger_isolated(self, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, console=False)

        app_logger = logging.getLogger('crest.app')
        metrics_logger = logging.getLogger('crest.metrics')

        app_logger.info('Application message')
        metrics_logger.info('Metrics message')

        # Check app log contains only app message
        app_log = Path(temp_log_dir) / 'crest.log.jsonl'
        with open(app_log) as f:
            content = f.read()
            assert 'Application message' in content
            # Metrics logger has propagate=False, so it doesn't appear in app log
            assert 'Metrics message' not in content

        # Check metrics log contains only metrics message
        metrics_log = Path(temp_log_dir) / 'metrics.jsonl'
        with open(metrics_log) as f:
            content = f.read()
            assert 'Metrics message' in content
            assert 'Application message' not in content


class TestClearHandlers:

    def test_clear_handlers_removes_duplicates(self, temp_log_dir):
        # Setup twice
        logger_setup(log_dir=temp_log_dir, console=False, clear_handlers=True)
        logger_setup(log_dir=temp_log_dir, console=False, clear_handlers=True)

        crest_logger = logging.getLogger('crest')
        # Should have only one file handler, not two
        file_handlers = [h for h in crest_logger.handlers
                         if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 2

    def test_clear_handlers_false_keeps_handlers(self, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, console=False, clear_handlers=True)
        initial_handler_count = len(logging.getLogger('crest').handlers)

        logger_setup(log_dir=temp_log_dir, console=False, clear_handlers=False)
        final_handler_count = len(logging.getLogger('crest').handlers)

        assert final_handler_count > initial_handler_count


class TestCrestJSONFormatter:

    def test_formatter_basic(self):
        formatter = CrestJSONFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test message',
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        log_entry = json.loads(output)

        assert log_entry['message'] == 'Test message'
        assert 'timestamp' in log_entry

    def test_formatter_custom_fields(self):
        formatter = CrestJSONFormatter(fmt_keys={
            'level': 'levelname',
            'module': 'module'
        })

        record = logging.LogRecord(
            name='test',
            level=logging.WARNING,
            pathname='',
            lineno=0,
            msg='Warning',
            args=(),
            exc_info=None
        )
        record.module = 'test_module'

        output = formatter.format(record)
        log_entry = json.loads(output)

        assert log_entry['level'] == 'WARNING'
        assert log_entry['module'] == 'test_module'

    def test_formatter_extra_attributes(self):
        formatter = CrestJSONFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test',
            args=(),
            exc_info=None
        )
        # Add custom attribute
        record.custom_field = 'custom_value'

        output = formatter.format(record)
        log_entry = json.loads(output)

        assert log_entry['custom_field'] == 'custom_value'


class TestLogLevelEnum:

    def test_enum_values(self):
        assert LogLevel.DEBUG.value == logging.DEBUG
        assert LogLevel.INFO.value == logging.INFO
        assert LogLevel.WARNING.value == logging.WARNING
        assert LogLevel.ERROR.value == logging.ERROR
        assert LogLevel.CRITICAL.value == logging.CRITICAL

    def test_enum_in_logger_setup(self, temp_log_dir):
        logger_setup(log_level=LogLevel.ERROR,
                     log_dir=temp_log_dir, console=False)

        crest_logger = logging.getLogger('crest')
        assert crest_logger.level == logging.ERROR
