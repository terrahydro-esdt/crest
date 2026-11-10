"""Checks that the stand-in loadEnv run log behaves like loadEnv's and cleans up after itself."""

import logging
import os
import sys

from ._fake_loadenv import fake_loadenv_run_log


def test_fixture_handler_is_on_root(fake_run_log, tmp_path):
    assert fake_run_log in logging.getLogger().handlers
    assert sys.modules["loadenv_logging"]._file_handler is fake_run_log
    assert fake_run_log.baseFilename == str(tmp_path / f"run-{os.getpid()}.log.jsonl")


def test_run_file_receives_records(fake_run_log):
    logging.getLogger("some.module").warning("to the run file")
    fake_run_log.flush()

    with open(fake_run_log.baseFilename) as f:
        assert "to the run file" in f.read()


def test_leaves_root_clean_afterwards(tmp_path):
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level
    module_before = sys.modules.get("loadenv_logging")

    with fake_loadenv_run_log(tmp_path) as handler:
        root.warning("opens the file")
        assert handler.stream is not None

    assert root.handlers == handlers_before
    assert root.level == level_before
    assert sys.modules.get("loadenv_logging") is module_before
    assert handler.stream is None  # closed
