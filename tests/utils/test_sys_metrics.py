"""
Tests for crest.utils.sys_metrics module.
"""

import logging
import re

import pytest

from crest.utils.sys_metrics import SysMetrics, _default_run_dir


class TestDefaultRunDir:
    """Tests for the _default_run_dir helper."""

    def test_default_base(self):
        run_dir = _default_run_dir()
        assert run_dir.startswith("logs/")

    def test_custom_base(self):
        run_dir = _default_run_dir(base="custom")
        assert run_dir.startswith("custom/")

    def test_timestamp_format(self):
        run_dir = _default_run_dir()
        timestamp = run_dir.split("/", 1)[1]
        assert re.fullmatch(r"\d{8}-\d{6}", timestamp)


class TestSysMetricsInit:
    """Tests for SysMetrics construction."""

    def test_no_args(self):
        """SysMetrics() with no arguments must work (see class docstring Example)."""
        sm = SysMetrics()
        assert sm.run_dir.startswith("logs/")
        assert sm.run_id == sm.run_dir.split("/", 1)[1]

    def test_explicit_run_id_and_dir(self, tmp_path):
        run_dir = str(tmp_path / "myrun")
        sm = SysMetrics(run_id="myrun-id", run_dir=run_dir)
        assert sm.run_id == "myrun-id"
        assert sm.run_dir == run_dir

    def test_run_id_defaults_to_run_dir_name(self, tmp_path):
        run_dir = str(tmp_path / "some-run")
        sm = SysMetrics(run_dir=run_dir)
        assert sm.run_id == "some-run"

    def test_has_process_handle(self):
        sm = SysMetrics()
        assert sm._proc.pid == __import__("os").getpid()


class TestSysMetricsEmit:
    """Tests for SysMetrics.emit().
    """

    def test_emit_does_not_raise(self):
        sm = SysMetrics(run_id="test", run_dir="logs/test")
        sm.emit("custom_event", foo="bar")

    def test_emit_logs_expected_record(self, caplog):
        sm = SysMetrics(run_id="test", run_dir="logs/test")
        with caplog.at_level(logging.INFO, logger="crest.metrics"):
            sm.emit("custom_event", foo="bar", count=3)

        records = [r for r in caplog.records if r.name == "crest.metrics"]
        assert len(records) == 1
        record = records[0]
        assert record.message == "custom_event"
        assert record.kind == "custom_event"
        assert record.foo == "bar"
        assert record.count == 3
        assert hasattr(record, "ts")


class TestSysMetricsTimed:
    """Tests for SysMetrics.timed()."""

    def test_timed_emits_expected_fields(self, caplog):
        sm = SysMetrics(run_id="test", run_dir="logs/test")
        with caplog.at_level(logging.INFO, logger="crest.metrics"):
            with sm.timed("my_label"):
                pass

        records = [r for r in caplog.records if r.name == "crest.metrics"]
        assert len(records) == 1
        record = records[0]
        assert record.kind == "timed"
        assert record.label == "my_label"
        assert record.elapsed >= 0
        assert hasattr(record, "rss_mb")
        assert hasattr(record, "rss_delta_mb")
        assert hasattr(record, "cpu_user_s")
        assert hasattr(record, "cpu_sys_s")

    def test_timed_still_emits_and_reraises_on_exception(self, caplog):
        sm = SysMetrics(run_id="test", run_dir="logs/test")
        with caplog.at_level(logging.INFO, logger="crest.metrics"):
            with pytest.raises(ValueError):
                with sm.timed("failing_label"):
                    raise ValueError("boom")

        records = [r for r in caplog.records if r.name == "crest.metrics"]
        assert len(records) == 1
        assert records[0].label == "failing_label"


class TestSysMetricsSnapshot:
    """Tests for SysMetrics.snapshot()."""

    def test_snapshot_emits_expected_fields(self, caplog):
        sm = SysMetrics(run_id="test", run_dir="logs/test")
        with caplog.at_level(logging.INFO, logger="crest.metrics"):
            sm.snapshot()

        records = [r for r in caplog.records if r.name == "crest.metrics"]
        assert len(records) == 1
        record = records[0]
        assert record.kind == "snapshot"
        assert hasattr(record, "rss_mb")
        assert hasattr(record, "vms_mb")
        assert hasattr(record, "threads")
        assert hasattr(record, "cpu_user_s")
        assert hasattr(record, "cpu_sys_s")
