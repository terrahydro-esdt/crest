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

    def test_initialize_only_once(self, temp_log_dir):
        """Test that initialize only runs once without force."""
        LoggingService.initialize(log_dir=temp_log_dir, log_level="DEBUG")
        config1 = LoggingService.get_config()

        # Second initialize should be ignored
        LoggingService.initialize(log_dir=temp_log_dir, log_level="WARNING")
        config2 = LoggingService.get_config()

        assert config1['log_level'] == 'DEBUG'
        assert config2['log_level'] == 'DEBUG'  # Unchanged

    def test_initialize_with_force(self, temp_log_dir):
        """Test that force=True allows reinitialization."""
        LoggingService.initialize(log_dir=temp_log_dir, log_level="DEBUG")
        LoggingService.initialize(log_dir=temp_log_dir, log_level="WARNING", force=True)

        config = LoggingService.get_config()
        assert config['log_level'] == 'WARNING'

    def test_get_logger_auto_initialize(self, temp_log_dir):
        """Test that get_logger auto-initializes if needed."""
        assert not LoggingService.is_initialized()

        logger = LoggingService.get_logger("crest.test")

        assert LoggingService.is_initialized()
        assert isinstance(logger, logging.Logger)

    def test_get_logger_adds_prefix(self, temp_log_dir):
        """Test that get_logger adds 'crest.' prefix if missing."""
        LoggingService.initialize(log_dir=temp_log_dir)

        logger = LoggingService.get_logger("mymodule")

        assert logger.name == "crest.mymodule"

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

        assert logger.name.startswith("crest")


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

        with log_context(run_id="exp_001"):
            logger.info("Inside context")

        logger.info("Outside context")

        log_file = Path(temp_log_dir) / "crest.log.jsonl"
        with open(log_file) as f:
            lines = f.readlines()

        inside = json.loads(lines[0])
        outside = json.loads(lines[1])

        assert 'run_id' in inside
        assert 'run_id' not in outside


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
