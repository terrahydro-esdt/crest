import pytest
import logging
import os
import json
import subprocess
import sys
import tempfile
import types
import io
import re
import datetime as dt
from pathlib import Path

from crest.utils import crest_logger
from crest.utils.crest_logger import (
    logger_setup,
    LogLevel,
    _find_run_log_handler,
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
            # scope="crest" sets propagate=False on "crest" - must be
            # restored, or every later scope="root" test silently stops
            # receiving messages logged through the "crest" hierarchy.
            logger.propagate = True
    # scope="root" (the default) attaches handlers to the root logger
    # instead of "crest", so it must be cleaned up too or state leaks
    # across tests.
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)  # logging module's own default
    # Levels/flags were restored by hand above, so forget what
    # logger_setup() recorded or a later reset would "restore" stale values.
    crest_logger._original_logger_state.clear()
    crest_logger._borrowed_handlers.clear()


@pytest.fixture(autouse=True)
def cleanup_env_vars():
    """Clean up after each test."""
    env_vars = [
        'CREST_LOG_LEVEL',
        'CREST_CONSOLE_LOG_LEVEL',
        'CREST_LOG_SCOPE',
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

        # Default scope="root": the root logger is configured, not "crest"
        # directly (see TestScope for scope-specific coverage).
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0

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

        assert logging.getLogger().level == logging.DEBUG

    def test_log_level_warning(self, temp_log_dir):
        logger_setup(log_level=LogLevel.WARNING,
                     log_dir=temp_log_dir, console=False)

        assert logging.getLogger().level == logging.WARNING

    def test_console_disabled(self, temp_log_dir):
        root_logger = logging.getLogger()
        # pytest's own log-capture plugin adds its own StreamHandlers to
        # root, so diff against a baseline to isolate what logger_setup
        # itself added, rather than filtering by handler type.
        before = list(root_logger.handlers)

        logger_setup(log_dir=temp_log_dir, console=False)

        added = [h for h in root_logger.handlers if h not in before]
        assert len(added) > 0  # the file handler(s) were added
        stream_handlers = [h for h in added
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


def _file_handlers(logger):
    """pytest's own log-capture plugin attaches handlers to the root
    logger, so tests must look for crest's own FileHandler specifically
    rather than asserting on raw handler counts."""
    return [h for h in logger.handlers if isinstance(h, logging.FileHandler)]


class TestScope:

    def test_default_scope_attaches_to_root_logger(self, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, console=False)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(_file_handlers(root_logger)) > 0
        # "crest" itself is untouched directly - it inherits via propagation
        assert logging.getLogger('crest').level == logging.NOTSET

    def test_default_scope_captures_unprefixed_module_logger(self, temp_log_dir):
        """Regression test for the data_metrics.py-style bug: a logger
        created the idiomatic way (plain module name, no "crest" prefix,
        no scope override) must still be captured when logging is set up
        with every parameter left at its default."""
        logger_setup(log_dir=temp_log_dir, console=False)

        logger = logging.getLogger('data_metrics')
        logger.info('metrics written')

        log_file = Path(temp_log_dir) / 'crest.log.jsonl'
        with open(log_file) as f:
            assert 'metrics written' in f.read()

    def test_root_scope_is_explicit_equivalent_of_default(self, temp_log_dir):
        logger_setup(scope='root', log_dir=temp_log_dir, console=False)

        assert logging.getLogger().level == logging.INFO
        assert len(_file_handlers(logging.getLogger())) > 0

    def test_crest_scope_preserves_isolation(self, temp_log_dir):
        logger_setup(scope='crest', log_dir=temp_log_dir, console=False)

        crest_logger = logging.getLogger('crest')
        assert crest_logger.level == logging.INFO
        assert crest_logger.propagate is False
        assert len(_file_handlers(crest_logger)) > 0
        # Root logger must gain no crest-configured file handler under scope="crest"
        assert len(_file_handlers(logging.getLogger())) == 0

    def test_root_scope_captures_unrelated_logger(self, temp_log_dir):
        logger_setup(scope='root', log_dir=temp_log_dir, console=False)

        logger = logging.getLogger('some_app.module')
        logger.info('from an unrelated package')

        log_file = Path(temp_log_dir) / 'crest.log.jsonl'
        with open(log_file) as f:
            assert 'from an unrelated package' in f.read()

    def test_crest_scope_does_not_capture_unrelated_logger(self, temp_log_dir):
        logger_setup(scope='crest', log_dir=temp_log_dir, console=False)

        logger = logging.getLogger('some_app.module')
        logger.info('from an unrelated package')

        log_file = Path(temp_log_dir) / 'crest.log.jsonl'
        with open(log_file) as f:
            assert 'from an unrelated package' not in f.read()

    def test_crest_scope_still_captures_crest_logger(self, temp_log_dir):
        logger_setup(scope='crest', log_dir=temp_log_dir, console=False)

        logger = logging.getLogger('crest.test')
        logger.info('from crest itself')

        log_file = Path(temp_log_dir) / 'crest.log.jsonl'
        with open(log_file) as f:
            assert 'from crest itself' in f.read()

    def test_invalid_scope_raises(self, temp_log_dir):
        with pytest.raises(ValueError, match="Invalid scope"):
            logger_setup(scope='invalid', log_dir=temp_log_dir, console=False)

    def test_env_scope(self, temp_log_dir):
        os.environ['CREST_LOG_SCOPE'] = 'crest'

        logger_setup(log_dir=temp_log_dir, console=False)

        assert len(_file_handlers(logging.getLogger('crest'))) > 0
        assert len(_file_handlers(logging.getLogger())) == 0

    def test_parameter_overrides_env_scope(self, temp_log_dir):
        os.environ['CREST_LOG_SCOPE'] = 'crest'

        logger_setup(scope='root', log_dir=temp_log_dir, console=False)

        assert len(_file_handlers(logging.getLogger())) > 0
        assert len(_file_handlers(logging.getLogger('crest'))) == 0


class TestEnvironmentVariables:

    def test_env_log_level(self, temp_log_dir):
        os.environ['CREST_LOG_LEVEL'] = 'DEBUG'

        logger_setup(log_dir=temp_log_dir, console=False)

        assert logging.getLogger().level == logging.DEBUG

    def test_env_console_level(self, temp_log_dir):
        os.environ['CREST_LOG_LEVEL'] = 'DEBUG'
        os.environ['CREST_CONSOLE_LOG_LEVEL'] = 'ERROR'

        root_logger = logging.getLogger()
        before = list(root_logger.handlers)

        logger_setup(log_dir=temp_log_dir, console=True)

        # Find the console handler among what logger_setup just added
        # (pytest's own log-capture plugin adds StreamHandlers to root too).
        added = [h for h in root_logger.handlers if h not in before]
        stream_handler = None
        for handler in added:
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

        assert logging.getLogger().level == logging.WARNING


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


class TestHandlerOwnership:

    def test_all_created_handlers_are_marked_owned(self, temp_log_dir):
        # pytest's own capture handlers may already sit on root; only look
        # at what logger_setup() added.
        before = list(logging.getLogger().handlers)
        logger_setup(log_dir=temp_log_dir, console=True)

        root_handlers = [h for h in logging.getLogger().handlers if h not in before]
        metrics_handlers = logging.getLogger('crest.metrics').handlers
        # app + error + console on root, metrics file on "crest.metrics"
        assert len(root_handlers) == 3
        assert len(metrics_handlers) == 1
        for handler in root_handlers + metrics_handlers:
            assert getattr(handler, '_crest_owned', False) is True

    def test_crest_scope_handlers_are_marked_owned(self, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, console=True, scope='crest')

        for handler in logging.getLogger('crest').handlers:
            assert getattr(handler, '_crest_owned', False) is True


class TestClearHandlers:

    def test_clear_handlers_removes_duplicates(self, temp_log_dir):
        # Setup twice
        logger_setup(log_dir=temp_log_dir, console=False, clear_handlers=True)
        logger_setup(log_dir=temp_log_dir, console=False, clear_handlers=True)

        # Should have only app + error file handlers (2), not 4, on the
        # default scope="root" target.
        file_handlers = _file_handlers(logging.getLogger())
        assert len(file_handlers) == 2

    def test_clear_handlers_false_keeps_handlers(self, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, console=False, clear_handlers=True)
        initial_handler_count = len(_file_handlers(logging.getLogger()))

        logger_setup(log_dir=temp_log_dir, console=False, clear_handlers=False)
        final_handler_count = len(_file_handlers(logging.getLogger()))

        assert final_handler_count > initial_handler_count


class TestForeignHandlers:

    @staticmethod
    def _foreign(logger):
        return [h for h in logger.handlers if not getattr(h, '_crest_owned', False)]

    def test_default_keeps_foreign_root_handler(self, temp_log_dir):
        root = logging.getLogger()
        foreign = logging.StreamHandler(io.StringIO())
        root.addHandler(foreign)
        foreign_before = self._foreign(root)

        logger_setup(log_dir=temp_log_dir, console=False)
        logger_setup(log_dir=temp_log_dir, console=False)

        # Still attached exactly once, and non-CREST handlers keep their order
        assert root.handlers.count(foreign) == 1
        assert self._foreign(root) == foreign_before
        # ...while CREST's own handlers were replaced, not duplicated
        assert len(_file_handlers(root)) == 2

    def test_foreign_root_handler_still_receives_records(self, temp_log_dir):
        stream = io.StringIO()
        foreign = logging.StreamHandler(stream)
        logging.getLogger().addHandler(foreign)

        logger_setup(log_dir=temp_log_dir, console=False)
        logging.getLogger('host.app').info('Host message')

        assert 'Host message' in stream.getvalue()

    def test_clear_all_removes_foreign_root_handler(self, temp_log_dir):
        root = logging.getLogger()
        foreign = logging.StreamHandler(io.StringIO())
        root.addHandler(foreign)

        logger_setup(log_dir=temp_log_dir, console=False, clear_handlers="all")

        assert foreign not in root.handlers
        assert all(getattr(h, '_crest_owned', False) for h in root.handlers)

    def test_clear_false_keeps_previous_crest_handlers(self, temp_log_dir):
        root = logging.getLogger()
        logger_setup(log_dir=temp_log_dir, console=False)
        first = [h for h in root.handlers if getattr(h, '_crest_owned', False)]

        logger_setup(log_dir=temp_log_dir, console=False, clear_handlers=False)

        assert all(h in root.handlers for h in first)

    @pytest.mark.parametrize('value', ['ALL', 'owned', 1.5, None])
    def test_invalid_clear_handlers_raises(self, temp_log_dir, value):
        with pytest.raises(ValueError, match='clear_handlers'):
            logger_setup(log_dir=temp_log_dir, console=False, clear_handlers=value)


class TestRootLevel:
    """CREST may lower the root logger's level, never raise it."""

    def test_quiet_crest_level_does_not_silence_host_handler(self, temp_log_dir):
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        stream = io.StringIO()
        host_handler = logging.StreamHandler(stream)
        root.addHandler(host_handler)

        logger_setup(log_level='WARNING', log_dir=temp_log_dir, console=False)
        logging.getLogger('host.app').info('host info message')

        assert root.level == logging.INFO
        assert 'host info message' in stream.getvalue()
        crest_log = (Path(temp_log_dir) / 'crest.log.jsonl').read_text()
        assert 'host info message' not in crest_log

    def test_verbose_crest_level_still_takes_effect(self, temp_log_dir):
        logger_setup(log_level='DEBUG', log_dir=temp_log_dir, console=False)
        logging.getLogger('host.app').debug('debug detail')

        assert logging.getLogger().level == logging.DEBUG
        crest_log = (Path(temp_log_dir) / 'crest.log.jsonl').read_text()
        assert 'debug detail' in crest_log

    def test_crest_scope_sets_exact_level_and_leaves_root(self, temp_log_dir):
        root_level_before = logging.getLogger().level

        logger_setup(scope='crest', log_level='WARNING', log_dir=temp_log_dir, console=False)

        assert logging.getLogger('crest').level == logging.WARNING
        assert logging.getLogger().level == root_level_before


class TestFindRunLogHandler:
    """Detecting loadEnv's run-file handler (see tests/utils/_fake_loadenv.py)."""

    @staticmethod
    def _install_module(monkeypatch, **attrs):
        module = types.ModuleType('loadenv_logging')
        for name, value in attrs.items():
            setattr(module, name, value)
        monkeypatch.setitem(sys.modules, 'loadenv_logging', module)
        return module

    def test_detects_fake_run_log(self, fake_run_log):
        assert _find_run_log_handler() is fake_run_log

    def test_no_module(self, monkeypatch):
        monkeypatch.delitem(sys.modules, 'loadenv_logging', raising=False)
        assert _find_run_log_handler() is None

    def test_file_handler_is_none(self, monkeypatch):
        # What loadEnv leaves when CREST_LOG_DIR is unset or the session is interactive.
        self._install_module(monkeypatch, _file_handler=None)
        assert _find_run_log_handler() is None

    def test_handler_not_on_root(self, monkeypatch, tmp_path):
        handler = logging.FileHandler(tmp_path / 'run.log.jsonl', delay=True)
        self._install_module(monkeypatch, _file_handler=handler)
        assert _find_run_log_handler() is None

    def test_public_accessor_preferred(self, monkeypatch, fake_run_log):
        stale = logging.NullHandler()
        module = sys.modules['loadenv_logging']
        monkeypatch.setattr(module, '_file_handler', stale, raising=False)
        monkeypatch.setattr(module, 'run_file_handler', lambda: fake_run_log, raising=False)
        assert _find_run_log_handler() is fake_run_log

    def test_failing_accessor_means_no_run_log(self, monkeypatch, fake_run_log):
        def broken():
            raise RuntimeError('boom')

        monkeypatch.setattr(sys.modules['loadenv_logging'], 'run_file_handler', broken,
                            raising=False)
        assert _find_run_log_handler() is None


class TestOptionalOutputs:
    """log_file, error_file and metrics_file can each be switched off with None."""

    def test_console_only_creates_no_files(self, tmp_path, capfd):
        log_dir = tmp_path / 'logs'
        logger_setup(log_dir=str(log_dir), log_file=None, error_file=None,
                     metrics_file=None, console=True)
        logging.getLogger('some.module').warning('console only')

        assert 'console only' in capfd.readouterr().err
        assert not log_dir.exists()

    def test_no_app_file_keeps_the_others(self, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, log_file=None, console=False)
        logging.getLogger('some.module').error('an error')

        assert not (Path(temp_log_dir) / 'crest.log.jsonl').exists()
        assert 'an error' in (Path(temp_log_dir) / 'error.jsonl').read_text()
        assert (Path(temp_log_dir) / 'metrics.jsonl').exists()

    def test_no_metrics_file_drops_metrics_quietly(self, temp_log_dir, capfd):
        logger_setup(log_dir=temp_log_dir, metrics_file=None, console=True)
        # WARNING, so logging's last-resort handler would print it if no handler were attached
        logging.getLogger('crest.metrics').warning('dropped metric')

        assert 'dropped metric' not in capfd.readouterr().err
        assert not (Path(temp_log_dir) / 'metrics.jsonl').exists()
        assert 'dropped metric' not in (Path(temp_log_dir) / 'crest.log.jsonl').read_text()

    def test_no_metrics_file_under_run_log_drops_metrics(self, fake_run_log, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, metrics_file=None, console=False)
        logging.getLogger('crest.metrics').info('dropped metric')
        fake_run_log.flush()

        assert fake_run_log not in logging.getLogger('crest.metrics').handlers
        run_file = Path(fake_run_log.baseFilename)
        assert not run_file.exists() or 'dropped metric' not in run_file.read_text()


class TestFileNamePlaceholders:
    """{pid} and {host} in file names give each process its own files."""

    def test_pid_in_every_file_name(self, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, console=False, log_file='app-{pid}.jsonl',
                     error_file='err-{pid}.jsonl', metrics_file='met-{pid}.jsonl')
        pid = os.getpid()

        for name in (f'app-{pid}.jsonl', f'err-{pid}.jsonl', f'met-{pid}.jsonl'):
            assert (Path(temp_log_dir) / name).exists()

    def test_host(self, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, console=False, log_file='app-{host}.jsonl')
        host = crest_logger._short_host()

        assert '.' not in host
        assert (Path(temp_log_dir) / f'app-{host}.jsonl').exists()

    @pytest.mark.parametrize('param', ['log_file', 'error_file', 'metrics_file'])
    @pytest.mark.parametrize('name', ['{foo}.jsonl', '{}.jsonl', 'app-{pid.jsonl'])
    def test_bad_placeholder_raises_naming_the_parameter(self, temp_log_dir, param, name):
        with pytest.raises(ValueError, match=param):
            logger_setup(log_dir=temp_log_dir, console=False, **{param: name})
        assert not any(Path(temp_log_dir).iterdir())

    def test_doubled_braces_are_literal(self, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, console=False, log_file='{{x}}-{pid}.jsonl')
        assert (Path(temp_log_dir) / f'{{x}}-{os.getpid()}.jsonl').exists()

    def test_two_processes_write_separate_files(self, tmp_path):
        # Loads crest_logger.py by path: it only needs the standard library, and importing
        # the crest package would take seconds per process.
        script = (
            'import importlib.util, logging, sys\n'
            'spec = importlib.util.spec_from_file_location("crest_logger", sys.argv[1])\n'
            'mod = importlib.util.module_from_spec(spec)\n'
            'spec.loader.exec_module(mod)\n'
            'mod.logger_setup(log_dir=sys.argv[2], log_file="app-{pid}.jsonl", console=False)\n'
            'logging.getLogger("worker").info("hello")\n'
            'print(mod.os.getpid())\n'
        )
        procs = [
            subprocess.Popen([sys.executable, '-c', script, crest_logger.__file__,
                              str(tmp_path)], stdout=subprocess.PIPE, text=True)
            for _ in range(2)
        ]
        pids = [int(p.communicate(timeout=60)[0]) for p in procs]
        assert all(p.returncode == 0 for p in procs)

        app_files = sorted(f.name for f in tmp_path.glob('app-*.jsonl'))
        assert app_files == sorted(f'app-{pid}.jsonl' for pid in pids)
        for name in app_files:
            assert 'hello' in (tmp_path / name).read_text()


class TestEnvLogFile:
    """CREST_LOG_FILE names CREST's app file only when there's no run log."""

    def test_used_without_run_log(self, temp_log_dir, monkeypatch):
        monkeypatch.setenv('CREST_LOG_FILE', 'env-{pid}.log.jsonl')
        logger_setup(log_dir=temp_log_dir, console=False)

        assert (Path(temp_log_dir) / f'env-{os.getpid()}.log.jsonl').exists()
        assert not (Path(temp_log_dir) / 'crest.log.jsonl').exists()

    def test_explicit_log_file_wins(self, temp_log_dir, monkeypatch):
        monkeypatch.setenv('CREST_LOG_FILE', 'env.log.jsonl')
        logger_setup(log_dir=temp_log_dir, console=False, log_file='mine.log.jsonl')

        assert (Path(temp_log_dir) / 'mine.log.jsonl').exists()
        assert not (Path(temp_log_dir) / 'env.log.jsonl').exists()

    def test_bad_placeholder_names_the_variable(self, temp_log_dir, monkeypatch):
        monkeypatch.setenv('CREST_LOG_FILE', '{foo}.log.jsonl')
        with pytest.raises(ValueError, match='CREST_LOG_FILE'):
            logger_setup(log_dir=temp_log_dir, console=False)

    def test_ignored_under_run_log(self, fake_run_log, temp_log_dir, monkeypatch):
        # loadEnv sets CREST_LOG_FILE to its own run file's name.
        monkeypatch.setenv('CREST_LOG_FILE', 'run-{pid}.log.jsonl')
        logger_setup(log_dir=temp_log_dir, console=False)
        logging.getLogger('some.module').info('only in the run file')
        fake_run_log.flush()

        assert sorted(f.name for f in Path(temp_log_dir).iterdir()) == [
            Path(fake_run_log.baseFilename).name
        ]
        assert Path(fake_run_log.baseFilename).read_text().count('only in the run file') == 1

    def test_ignored_under_run_log_with_ignore(self, fake_run_log, temp_log_dir, monkeypatch):
        monkeypatch.setenv('CREST_LOG_FILE', 'run-{pid}.log.jsonl')
        logger_setup(log_dir=temp_log_dir, console=False, run_log='ignore')
        logging.getLogger('some.module').info('once per file')
        fake_run_log.flush()

        app_handlers = [h for h in _file_handlers(logging.getLogger())
                        if getattr(h, '_crest_owned', False) and h.level != logging.ERROR]
        assert [h.baseFilename for h in app_handlers] == [
            str(Path(temp_log_dir) / 'crest.log.jsonl')
        ]
        assert Path(fake_run_log.baseFilename).read_text().count('once per file') == 1


class TestRunLog:
    """Under loadEnv, records go to the run file once and CREST skips its own app/error files."""

    @staticmethod
    def _run_file_lines(handler):
        handler.flush()
        path = Path(handler.baseFilename)
        # delay=True: the file only exists once a record has reached it.
        return path.read_text().splitlines() if path.exists() else []

    def test_default_writes_each_record_once_to_run_file(self, fake_run_log, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, console=False)
        logging.getLogger('some.module').info('once only')

        assert self._run_file_lines(fake_run_log).count('once only') == 1
        assert not (Path(temp_log_dir) / 'crest.log.jsonl').exists()
        assert not (Path(temp_log_dir) / 'error.jsonl').exists()

    def test_console_still_prints(self, fake_run_log, temp_log_dir, capfd):
        logger_setup(log_dir=temp_log_dir, console=True, console_level='WARNING')
        logging.getLogger('some.module').info('quiet on console')
        logging.getLogger('some.module').warning('loud on console')

        err = capfd.readouterr().err
        assert 'loud on console' in err
        assert 'quiet on console' not in err
        assert err.count('loud on console') == 1

    def test_ignore_writes_crest_files_too(self, fake_run_log, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, console=False, run_log='ignore')
        logging.getLogger('some.module').info('in both')

        assert self._run_file_lines(fake_run_log).count('in both') == 1
        assert 'in both' in (Path(temp_log_dir) / 'crest.log.jsonl').read_text()

    def test_without_run_log_writes_crest_files(self, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, console=False)
        logging.getLogger('some.module').info('no run log')

        assert 'no run log' in (Path(temp_log_dir) / 'crest.log.jsonl').read_text()

    def test_crest_scope_writes_crest_files(self, fake_run_log, temp_log_dir):
        # "crest" doesn't propagate to root, so the run file would never see these records.
        logger_setup(log_dir=temp_log_dir, console=False, scope='crest')
        logging.getLogger('crest.app').info('isolated')

        assert 'isolated' in (Path(temp_log_dir) / 'crest.log.jsonl').read_text()
        assert 'isolated' not in self._run_file_lines(fake_run_log)

    def test_clear_all_removes_run_log_so_crest_writes_files(self, fake_run_log, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, console=False, clear_handlers='all')

        assert fake_run_log not in logging.getLogger().handlers
        assert (Path(temp_log_dir) / 'crest.log.jsonl').exists()

    def test_invalid_run_log_raises(self, temp_log_dir):
        with pytest.raises(ValueError, match='run_log'):
            logger_setup(log_dir=temp_log_dir, run_log='bogus')

    def test_metrics_go_to_run_file_not_console(self, fake_run_log, temp_log_dir, capfd):
        fake_run_log.setFormatter(logging.Formatter('%(name)s: %(message)s'))
        logger_setup(log_dir=temp_log_dir, console=True)
        logging.getLogger('crest.metrics').info('epoch=1 loss=0.5')

        assert self._run_file_lines(fake_run_log).count('crest.metrics: epoch=1 loss=0.5') == 1
        assert not (Path(temp_log_dir) / 'metrics.jsonl').exists()
        assert 'epoch=1' not in capfd.readouterr().err

    def test_run_file_handler_is_borrowed_not_owned(self, fake_run_log, temp_log_dir):
        logger_setup(log_dir=temp_log_dir, console=False)

        assert fake_run_log in logging.getLogger('crest.metrics').handlers
        assert not getattr(fake_run_log, '_crest_owned', False)

    def test_switching_to_ignore_detaches_run_file_from_metrics(self, fake_run_log,
                                                                 temp_log_dir):
        logger_setup(log_dir=temp_log_dir, console=False)
        logger_setup(log_dir=temp_log_dir, console=False, run_log='ignore')
        logging.getLogger('crest.metrics').info('to metrics.jsonl')

        assert fake_run_log not in logging.getLogger('crest.metrics').handlers
        assert fake_run_log in logging.getLogger().handlers
        assert 'to metrics.jsonl' in (Path(temp_log_dir) / 'metrics.jsonl').read_text()
        assert 'to metrics.jsonl' not in self._run_file_lines(fake_run_log)


class TestRemovedHandlersAreClosed:

    @staticmethod
    def _owned_file_handlers():
        loggers = [logging.getLogger(), logging.getLogger('crest.metrics')]
        return [
            h for lg in loggers for h in lg.handlers
            if isinstance(h, logging.FileHandler) and getattr(h, '_crest_owned', False)
        ]

    def test_reconfiguration_closes_removed_file_handlers(self, temp_log_dir):
        removed = []
        logger_setup(log_dir=temp_log_dir, console=False)
        for _ in range(20):
            removed.extend(self._owned_file_handlers())
            logger_setup(log_dir=temp_log_dir, console=False)

        # app + error + metrics per reconfiguration
        assert len(removed) == 60
        current = self._owned_file_handlers()
        for handler in removed:
            assert handler not in current
            assert handler.stream is None  # FileHandler.close() drops the stream

    @pytest.mark.skipif(not sys.platform.startswith('linux'), reason='needs /proc/self/fd')
    def test_reconfiguration_does_not_leak_file_descriptors(self, temp_log_dir):
        def open_log_fds():
            fds = []
            for fd in os.listdir('/proc/self/fd'):
                try:
                    target = os.readlink(f'/proc/self/fd/{fd}')
                except OSError:
                    continue
                if target.startswith(temp_log_dir):
                    fds.append(target)
            return len(fds)

        logger_setup(log_dir=temp_log_dir, console=False)
        baseline = open_log_fds()
        for _ in range(20):
            logger_setup(log_dir=temp_log_dir, console=False)

        assert open_log_fds() == baseline


class TestTimestamps:
    """JSON timestamps follow CREST_LOG_TZ, with millisecond precision, like loadEnv's."""

    # 2026-01-15 12:00:00.123456 UTC: winter, so New York is at -05:00.
    CREATED = dt.datetime(2026, 1, 15, 12, 0, 0, 123456, tzinfo=dt.timezone.utc).timestamp()

    def _timestamp(self):
        record = logging.LogRecord('test', logging.INFO, '', 0, 'msg', (), None)
        record.created = self.CREATED
        return json.loads(CrestJSONFormatter().format(record))['timestamp']

    def test_utc(self, monkeypatch):
        monkeypatch.setenv('CREST_LOG_TZ', 'UTC')
        assert self._timestamp() == '2026-01-15T12:00:00.123+00:00'

    def test_default_is_new_york(self):
        assert self._timestamp() == '2026-01-15T07:00:00.123-05:00'

    def test_millisecond_precision(self, monkeypatch):
        monkeypatch.setenv('CREST_LOG_TZ', 'Asia/Kolkata')
        assert re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+05:30', self._timestamp())

    def test_invalid_zone_falls_back_to_local_time(self, monkeypatch):
        monkeypatch.setenv('CREST_LOG_TZ', 'Not/AZone')
        stamp = dt.datetime.fromisoformat(self._timestamp())

        assert stamp.utcoffset() is not None
        assert stamp == dt.datetime.fromtimestamp(self.CREATED, dt.timezone.utc).replace(
            microsecond=123000)

    def test_text_format_uses_the_same_time(self, monkeypatch):
        monkeypatch.setenv('CREST_LOG_TZ', 'UTC')
        record = logging.LogRecord('some.module', logging.INFO, '', 0, 'msg', (), None)
        record.created = self.CREATED

        assert crest_logger._IsoTextFormatter('%(asctime)s %(message)s').format(record) == (
            '2026-01-15T12:00:00.123+00:00 msg'
        )

    def test_text_line_written_by_logger_setup(self, temp_log_dir, monkeypatch):
        monkeypatch.setenv('CREST_LOG_TZ', 'Asia/Kolkata')
        logger_setup(log_dir=temp_log_dir, console=False, format_type='text')
        logging.getLogger('some.module').info('stamped')

        line = (Path(temp_log_dir) / 'crest.log.jsonl').read_text().splitlines()[-1]
        assert re.fullmatch(
            r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+05:30 - some\.module - INFO - stamped', line
        )

    def test_written_by_logger_setup(self, temp_log_dir, monkeypatch):
        monkeypatch.setenv('CREST_LOG_TZ', 'UTC')
        logger_setup(log_dir=temp_log_dir, console=False)
        logging.getLogger('some.module').info('stamped')

        line = (Path(temp_log_dir) / 'crest.log.jsonl').read_text().splitlines()[-1]
        assert re.search(r'\.\d{3}\+00:00$', json.loads(line)['timestamp'])


class TestIdentityFields:
    """JSON records carry host, pid and run_id like loadEnv's, without overwriting them."""

    @staticmethod
    def _format(**extra):
        record = logging.getLogger('some.module').makeRecord(
            'some.module', logging.INFO, '', 0, 'msg', (), None, extra=extra or None)
        return json.loads(CrestJSONFormatter().format(record))

    def test_added_when_missing(self, monkeypatch):
        monkeypatch.setenv('CREST_RUN_ID', 'r1')
        entry = self._format()

        assert entry['run_id'] == 'r1'
        assert entry['pid'] == os.getpid()
        assert entry['host'] == crest_logger._short_host()

    def test_no_run_id_without_env(self):
        assert 'run_id' not in self._format()

    def test_stage_and_slurm_ids_from_env(self, monkeypatch):
        monkeypatch.setenv('CREST_STAGE', 'ingest')
        monkeypatch.setenv('SLURM_JOB_ID', '4242')
        monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '7')
        entry = self._format()

        assert (entry['stage'], entry['slurm_job_id'], entry['slurm_array_task_id']) == (
            'ingest', '4242', '7')

    def test_record_stage_wins_over_env(self, monkeypatch):
        monkeypatch.setenv('CREST_STAGE', 'ingest')
        assert self._format(stage='training')['stage'] == 'training'

    def test_existing_values_are_kept(self, monkeypatch):
        monkeypatch.setenv('CREST_RUN_ID', 'r1')
        entry = self._format(host='h', pid=1, run_id='x')

        assert (entry['host'], entry['pid'], entry['run_id']) == ('h', 1, 'x')

    def test_written_by_logger_setup(self, temp_log_dir, monkeypatch):
        monkeypatch.setenv('CREST_RUN_ID', 'r1')
        logger_setup(log_dir=temp_log_dir, console=False)
        logging.getLogger('some.module').info('identified')

        entry = json.loads((Path(temp_log_dir) / 'crest.log.jsonl').read_text().splitlines()[-1])
        assert (entry['run_id'], entry['pid']) == ('r1', os.getpid())


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

        # The enum level lands on CREST's own app-file handler; root is only
        # ever lowered, so it stays at its default (WARNING).
        app_handler = next(h for h in _file_handlers(logging.getLogger())
                           if h.baseFilename.endswith('crest.log.jsonl'))
        assert app_handler.level == logging.ERROR
        assert logging.getLogger().level == logging.WARNING
