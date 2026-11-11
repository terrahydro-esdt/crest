"""
Tests for crest.utils.logging_service module (Phase 3 features).

This module tests the advanced logging features including:
- LoggingService centralized management
- Context-aware logging
- Function decorators
- Helper utilities
"""

import pytest
import logging
import os
import json
import time
import io
from pathlib import Path

from crest.utils.logging_service import (
    LoggingService,
    get_logger,
    log_context,
    log_function_call,
    log_execution_time,
)


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create a temporary directory for log files."""
    return str(tmp_path)


def _owned(logger):
    """Handlers CREST attached to `logger`."""
    return [h for h in logger.handlers if getattr(h, "_crest_owned", False)]


@pytest.fixture(autouse=True)
def cleanup_logging_service():
    """Reset LoggingService after each test."""
    yield
    LoggingService.reset()


class TestLoggingService:
    """Tests for LoggingService class."""

    def test_initialize_basic(self, temp_log_dir):
        """Test basic LoggingService initialization."""
        LoggingService.initialize(log_dir=temp_log_dir, console=False)

        assert LoggingService.is_initialized()
        assert Path(temp_log_dir, "crest.log.jsonl").exists()

    def test_initialize_accepts_rotate_kwargs(self, temp_log_dir):
        """LoggingService.initialize() must forward rotate/max_bytes/
        backup_count to logger_setup() - these previously raised
        TypeError because they weren't in its signature at all."""
        from logging.handlers import RotatingFileHandler

        LoggingService.initialize(
            log_dir=temp_log_dir, console=False,
            rotate=True, max_bytes=1234, backup_count=2,
        )

        file_handlers = [h for h in logging.getLogger().handlers
                         if isinstance(h, RotatingFileHandler)]
        assert len(file_handlers) > 0
        assert file_handlers[0].maxBytes == 1234
        assert file_handlers[0].backupCount == 2

    def test_initialize_accepts_error_file(self, temp_log_dir):
        """LoggingService.initialize() must forward error_file too -
        previously only logger_setup()'s own default ("error.jsonl") was
        ever usable; a custom name silently had no effect."""
        LoggingService.initialize(
            log_dir=temp_log_dir, console=False, error_file="custom_errors.jsonl",
        )

        assert Path(temp_log_dir, "custom_errors.jsonl").exists()

    def test_initialize_only_once(self, temp_log_dir):
        """Test that initialize only runs once without force."""
        LoggingService.initialize(log_dir=temp_log_dir, log_level="DEBUG")
        config1 = LoggingService.get_config()

        # Second initialize should be ignored
        LoggingService.initialize(log_dir=temp_log_dir, log_level="WARNING")
        config2 = LoggingService.get_config()

        assert config1['log_level'] == 'DEBUG'
        assert config2['log_level'] == 'DEBUG'  # Unchanged

    def test_initialize_only_once_keeps_same_handler_objects(self, temp_log_dir):
        """A second initialize() without force=True must be a true no-op -
        not just unchanged config, but the exact same handler objects left
        attached (no teardown/rebuild, no duplication)."""
        LoggingService.initialize(log_dir=temp_log_dir, console=False)
        handlers_after_first = list(logging.getLogger().handlers)

        LoggingService.initialize(log_dir=temp_log_dir, console=False, log_level="DEBUG")
        handlers_after_second = list(logging.getLogger().handlers)

        assert handlers_after_second == handlers_after_first

    def test_initialize_with_force(self, temp_log_dir):
        """Test that force=True allows reinitialization, rebuilding CREST's
        handlers while leaving a foreign root handler attached."""
        root = logging.getLogger()
        foreign = logging.StreamHandler(io.StringIO())
        root.addHandler(foreign)

        LoggingService.initialize(log_dir=temp_log_dir, console=False, log_level="DEBUG")
        owned_before = _owned(root)

        LoggingService.initialize(log_dir=temp_log_dir, console=False, log_level="WARNING", force=True)
        owned_after = _owned(root)

        config = LoggingService.get_config()
        assert config['log_level'] == 'WARNING'
        assert owned_before and owned_after
        assert not set(owned_before) & set(owned_after)  # replaced, not kept
        assert root.handlers.count(foreign) == 1
        root.removeHandler(foreign)

    def test_get_logger_auto_initialize(self, temp_log_dir):
        """Test that get_logger auto-initializes if needed."""
        assert not LoggingService.is_initialized()

        logger = LoggingService.get_logger("crest.test")

        assert LoggingService.is_initialized()
        assert isinstance(logger, logging.Logger)

    def test_get_logger_preserves_name(self, temp_log_dir):
        """get_logger must not rewrite or prefix the caller's logger name -
        under the default scope="root", any name is captured regardless."""
        LoggingService.initialize(log_dir=temp_log_dir)

        logger = LoggingService.get_logger("mymodule")

        assert logger.name == "mymodule"

    def test_reconfigure_keeps_foreign_root_handler(self, temp_log_dir):
        root = logging.getLogger()
        stream = io.StringIO()
        foreign = logging.StreamHandler(stream)
        root.addHandler(foreign)

        LoggingService.initialize(log_dir=temp_log_dir, console=False)
        LoggingService.reconfigure(log_level="DEBUG")
        logging.getLogger("host.app").info("after reconfigure")

        assert root.handlers.count(foreign) == 1
        assert "after reconfigure" in stream.getvalue()
        root.removeHandler(foreign)

    def test_reconfigure_to_quieter_level_restores_root_level(self, temp_log_dir):
        """Going DEBUG -> WARNING must not leave root at DEBUG: root's level is
        computed from what it was before CREST first changed it."""
        root_level_before = logging.getLogger().level  # WARNING by default

        LoggingService.initialize(log_dir=temp_log_dir, console=False, log_level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG

        LoggingService.reconfigure(log_level="WARNING")
        assert logging.getLogger().level == root_level_before

    def test_reconfigure(self, temp_log_dir):
        """Test reconfiguring logging at runtime."""
        LoggingService.initialize(log_dir=temp_log_dir, log_level="INFO")

        LoggingService.reconfigure(log_level="DEBUG")

        config = LoggingService.get_config()
        assert config['log_level'] == 'DEBUG'

    def test_get_config(self, temp_log_dir):
        """Test getting current configuration."""
        LoggingService.initialize(
            log_dir=temp_log_dir,
            log_level="INFO",
            format_type="text"
        )

        config = LoggingService.get_config()

        assert config['log_level'] == 'INFO'
        assert config['format_type'] == 'text'
        assert config['log_dir'] == temp_log_dir

    def test_reset(self, temp_log_dir):
        """Test resetting LoggingService."""
        LoggingService.initialize(log_dir=temp_log_dir)
        assert LoggingService.is_initialized()

        LoggingService.reset()

        assert not LoggingService.is_initialized()
        assert LoggingService.get_config() == {}

    def test_reset_leaves_foreign_handlers_and_restores_state(self, temp_log_dir):
        root = logging.getLogger()
        foreign = logging.StreamHandler(io.StringIO())
        root.addHandler(foreign)
        root_level_before = root.level

        LoggingService.initialize(
            log_dir=temp_log_dir, console=False, log_level="DEBUG",
            module_levels={"tests.dummy": "DEBUG"},
        )
        assert logging.getLogger("tests.dummy").level == logging.DEBUG
        assert logging.getLogger("crest.metrics").propagate is False

        LoggingService.reset()

        assert root.handlers.count(foreign) == 1
        all_loggers = [root] + [
            lg for lg in logging.Logger.manager.loggerDict.values()
            if isinstance(lg, logging.Logger)
        ]
        assert not any(_owned(lg) for lg in all_loggers)
        assert logging.getLogger("tests.dummy").level == logging.NOTSET
        assert root.level == root_level_before
        assert logging.getLogger("crest.metrics").propagate is True
        root.removeHandler(foreign)

    def test_reset_closes_crest_file_handlers(self, temp_log_dir):
        LoggingService.initialize(log_dir=temp_log_dir, console=False)
        file_handlers = [
            h for lg in (logging.getLogger(), logging.getLogger("crest.metrics"))
            for h in _owned(lg) if isinstance(h, logging.FileHandler)
        ]
        assert file_handlers

        LoggingService.reset()

        assert all(h.stream is None for h in file_handlers)

    def test_caplog_keeps_capturing_after_initialize(self, temp_log_dir, caplog):
        """pytest's capture handler lives on the root logger; initializing
        CREST logging inside a test must not remove it."""
        LoggingService.initialize(log_dir=temp_log_dir, console=False)

        logging.getLogger("host.app").info("captured after init")

        assert any(r.getMessage() == "captured after init" for r in caplog.records)


class TestRunLogOption:

    def test_run_log_is_stored_and_kept_on_reconfigure(self, temp_log_dir):
        LoggingService.initialize(log_dir=temp_log_dir, console=False, run_log="ignore")
        assert LoggingService.get_config()["run_log"] == "ignore"

        LoggingService.reconfigure(log_level="DEBUG")
        assert LoggingService.get_config()["run_log"] == "ignore"

    def test_default_skips_crest_files_under_run_log(self, fake_run_log, temp_log_dir):
        LoggingService.initialize(log_dir=temp_log_dir, console=False)
        get_logger("some.module").info("via the service")
        fake_run_log.flush()

        assert "via the service" in Path(fake_run_log.baseFilename).read_text()
        assert not Path(temp_log_dir, "crest.log.jsonl").exists()

    def test_invalid_run_log_raises(self, temp_log_dir):
        with pytest.raises(ValueError, match="run_log"):
            LoggingService.initialize(log_dir=temp_log_dir, run_log="bogus")

    def test_reset_leaves_run_file_open_and_on_root(self, fake_run_log, temp_log_dir):
        LoggingService.initialize(log_dir=temp_log_dir, console=False)
        get_logger("crest.metrics").info("opens the run file")
        assert fake_run_log.stream is not None

        LoggingService.reset()

        assert fake_run_log not in logging.getLogger("crest.metrics").handlers
        assert fake_run_log in logging.getLogger().handlers
        # Checked before any new write: FileHandler.emit() would reopen a closed handler.
        assert fake_run_log.stream is not None

    def test_force_twice_attaches_run_file_to_metrics_once(self, fake_run_log, temp_log_dir):
        LoggingService.initialize(log_dir=temp_log_dir, console=False, force=True)
        LoggingService.initialize(log_dir=temp_log_dir, console=False, force=True)

        metrics_handlers = logging.getLogger("crest.metrics").handlers
        assert metrics_handlers.count(fake_run_log) == 1
        assert logging.getLogger().handlers.count(fake_run_log) == 1


class TestGetLogger:
    """Tests for get_logger convenience function."""

    def test_get_logger_explicit_name(self, temp_log_dir):
        """Test get_logger with explicit name."""
        LoggingService.initialize(log_dir=temp_log_dir, console=False)

        logger = get_logger("crest.test")

        assert logger.name == "crest.test"

    def test_get_logger_auto_name(self, temp_log_dir):
        """Test get_logger with auto-detected name."""
        LoggingService.initialize(log_dir=temp_log_dir, console=False)

        logger = get_logger()  # Should detect __name__ from caller

        # Exactly the caller's module name - no "crest." prefix is added
        assert logger.name == __name__


class TestLogContext:
    """Tests for log_context context manager."""

    def test_log_context_adds_fields(self, temp_log_dir):
        """Test that log_context adds fields to log records."""
        LoggingService.initialize(
            log_dir=temp_log_dir,
            console=False,
            format_type="json"
        )
        logger = get_logger("crest.test")

        with log_context(run_id="exp_001", stage="training"):
            logger.info("Test message")

        # Read log file and verify extra fields
        log_file = Path(temp_log_dir) / "crest.log.jsonl"
        with open(log_file) as f:
            line = f.readline()
            log_entry = json.loads(line)

        assert log_entry['run_id'] == "exp_001"
        assert log_entry['stage'] == "training"
        assert log_entry['message'] == "Test message"

    def test_log_context_nested(self, temp_log_dir):
        """Test nested log_context managers."""
        LoggingService.initialize(
            log_dir=temp_log_dir,
            console=False,
            format_type="json"
        )
        logger = get_logger("crest.test")

        with log_context(run_id="exp_001"):
            with log_context(stage="training"):
                logger.info("Nested context")

        log_file = Path(temp_log_dir) / "crest.log.jsonl"
        with open(log_file) as f:
            line = f.readline()
            log_entry = json.loads(line)

        # Should have both fields from nested contexts
        assert log_entry['run_id'] == "exp_001"
        assert log_entry['stage'] == "training"

    def test_log_context_cleanup(self, temp_log_dir):
        """Test that log_context cleans up after exiting."""
        LoggingService.initialize(
            log_dir=temp_log_dir,
            console=False,
            format_type="json"
        )
        logger = get_logger("crest.test")

        # "run", not "run_id": loadEnv stamps run_id on every record, and so does CREST's
        # JSON formatter when CREST_RUN_ID is set.
        with log_context(run="exp_001"):
            logger.info("Inside context")

        logger.info("Outside context")

        log_file = Path(temp_log_dir) / "crest.log.jsonl"
        with open(log_file) as f:
            lines = f.readlines()

        inside = json.loads(lines[0])
        outside = json.loads(lines[1])

        assert 'run' in inside
        assert 'run' not in outside


class TestLogFunctionCall:
    """Tests for log_function_call decorator."""

    def test_log_function_call_basic(self, temp_log_dir):
        """Test basic function call logging."""
        LoggingService.initialize(log_dir=temp_log_dir, console=False)
        logger = get_logger("crest.test")

        @log_function_call(logger=logger, level="INFO")
        def test_func(x, y):
            return x + y

        result = test_func(2, 3)

        assert result == 5

        # Check logs
        log_file = Path(temp_log_dir) / "crest.log.jsonl"
        with open(log_file) as f:
            logs = f.readlines()

        assert len(logs) >= 2  # Entry and exit logs
        assert "Calling test_func" in logs[0]
        assert "returned 5" in logs[1]

    def test_log_function_call_with_exception(self, temp_log_dir):
        """Test function call logging with exception."""
        LoggingService.initialize(log_dir=temp_log_dir, console=False)
        logger = get_logger("crest.test")

        @log_function_call(logger=logger)
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_func()

        # Check logs contain exception info
        log_file = Path(temp_log_dir) / "crest.log.jsonl"
        with open(log_file) as f:
            content = f.read()

        assert "ValueError" in content
        assert "failing_func" in content


class TestLogExecutionTime:
    """Tests for log_execution_time decorator."""

    def test_log_execution_time_basic(self, temp_log_dir):
        """Test execution time logging."""
        LoggingService.initialize(
            log_dir=temp_log_dir,
            console=False,
            format_type="json"
        )
        logger = get_logger("crest.test")

        @log_execution_time(logger=logger, level="INFO")
        def slow_func():
            time.sleep(0.1)
            return "done"

        result = slow_func()

        assert result == "done"

        # Check logs
        log_file = Path(temp_log_dir) / "crest.log.jsonl"
        with open(log_file) as f:
            line = f.readline()
            log_entry = json.loads(line)

        assert 'duration_sec' in log_entry
        assert log_entry['duration_sec'] >= 0.1
        assert 'function' in log_entry
        assert log_entry['function'] == 'slow_func'

    def test_log_execution_time_with_exception(self, temp_log_dir):
        """Test execution time logging with exception."""
        LoggingService.initialize(
            log_dir=temp_log_dir,
            console=False,
            format_type="json"
        )
        logger = get_logger("crest.test")

        @log_execution_time(logger=logger)
        def failing_func():
            time.sleep(0.05)
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError):
            failing_func()

        # Check logs contain duration even on failure
        log_file = Path(temp_log_dir) / "crest.log.jsonl"
        with open(log_file) as f:
            line = f.readline()
            log_entry = json.loads(line)

        assert 'duration_sec' in log_entry
        assert log_entry['duration_sec'] >= 0.05


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_combined_features(self, temp_log_dir):
        """Test using multiple logging features together."""
        LoggingService.initialize(
            log_dir=temp_log_dir,
            console=False,
            format_type="json",
            log_level="DEBUG"
        )
        logger = get_logger("crest.integration")

        @log_execution_time(logger=logger)
        @log_function_call(logger=logger)
        def process_batch(batch_id, size):
            time.sleep(0.05)
            return {"batch_id": batch_id, "items": size}

        with log_context(run_id="exp_001", stage="processing"):
            result = process_batch("batch_1", 100)

        assert result == {"batch_id": "batch_1", "items": 100}

        # Verify all features worked
        log_file = Path(temp_log_dir) / "crest.log.jsonl"
        with open(log_file) as f:
            logs = [json.loads(line) for line in f]

        # Should have logs from both decorators
        assert len(logs) >= 3

        # Filter out setup logs, focus on app logs
        app_logs = [log for log in logs if log.get('logger') == 'crest.integration']

        # All app logs should have context
        assert len(app_logs) >= 3
        for log in app_logs:
            assert log.get('run_id') == "exp_001"
            assert log.get('stage') == "processing"

        # At least one log should have duration
        assert any('duration_sec' in log for log in app_logs)
