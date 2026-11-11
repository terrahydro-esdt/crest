"""
A stand-in for ast_utils' ``loadenv_logging`` module, for tests.

Under loadEnv, ``sitecustomize`` attaches one run-file handler to the root logger and keeps it
in ``loadenv_logging._file_handler``. CREST detects that handler through ``sys.modules`` only,
so tests can fake it without ast_utils being installed.
"""

import contextlib
import logging
import os
import sys
import types
from pathlib import Path
from typing import Iterator

import pytest


@contextlib.contextmanager
def fake_loadenv_run_log(log_dir: Path) -> Iterator[logging.FileHandler]:
    """Install a fake ``loadenv_logging`` run log on the root logger.

    Parameters
    ----------
    log_dir : Path
        Folder for the run file, ``run-{pid}.log.jsonl``.

    Yields
    ------
    logging.FileHandler
        The run-file handler, attached to root and stored as the fake module's
        ``_file_handler``. On exit it is removed from root and closed, and ``sys.modules`` is
        restored.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as d:
    ...     with fake_loadenv_run_log(Path(d)) as handler:
    ...         handler in logging.getLogger().handlers
    True
    """
    module = types.ModuleType("loadenv_logging")
    # delay=True, like loadEnv's RunFileHandler: the file is created on the first record.
    module._file_handler = logging.FileHandler(
        Path(log_dir) / f"run-{os.getpid()}.log.jsonl", delay=True
    )
    root = logging.getLogger()
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "loadenv_logging", module)
        root.addHandler(module._file_handler)
        try:
            yield module._file_handler
        finally:
            root.removeHandler(module._file_handler)
            module._file_handler.close()
