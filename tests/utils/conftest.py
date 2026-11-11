"""Fixtures shared by the tests in ``tests/utils``."""

import sys

import pytest

from ._fake_loadenv import fake_loadenv_run_log


# Variables a loadEnv shell, a loadEnv run or a Slurm job may set that change what CREST's
# logger_setup() does or what its records contain.
LOADENV_VARIABLES = (
    "CREST_LOG_LEVEL",
    "CREST_CONSOLE_LOG_LEVEL",
    "CREST_LOG_FORMAT",
    "CREST_LOG_DIR",
    "CREST_LOG_FILE",
    "CREST_LOG_SCOPE",
    "CREST_LOG_TZ",
    "CREST_RUN_ID",
    "CREST_STAGE",
    "SLURM_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
)


@pytest.fixture(autouse=True)
def no_real_run_log(monkeypatch):
    """Hide a real loadEnv run log and loadEnv's settings, so tests behave the same inside and
    outside a loadEnv shell.

    With ``CREST_LOG_AUTO=1``, ast_utils' ``sitecustomize`` attaches a run log to pytest's own
    process, and CREST would then skip the files many tests expect. Tests that need one of these
    variables set it themselves.
    """
    monkeypatch.delitem(sys.modules, "loadenv_logging", raising=False)
    for name in LOADENV_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_run_log(tmp_path):
    """A stand-in loadEnv run log on the root logger, removed and closed afterwards."""
    with fake_loadenv_run_log(tmp_path) as handler:
        yield handler
