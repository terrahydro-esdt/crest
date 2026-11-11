"""log_context(): no collision with ``extra``, per-thread/per-task isolation, and precedence over
loadEnv-style stamping."""

import asyncio
import inspect
import logging
import threading

import pytest

from crest.utils import logging_service
from crest.utils.logging_service import _install_context_stamping, log_context


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def captured():
    """Records logged to the "ctx.test" logger, kept away from root."""
    logger = logging.getLogger("ctx.test")
    handler = ListHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    yield logger, handler.records
    logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def _loadenv_style(original):
    """Like loadenv_logging's patch: after makeRecord, fill fields only where missing."""

    def make_record(self, *args, **kwargs):
        record = original(self, *args, **kwargs)
        if not hasattr(record, "stage"):
            record.stage = "ingest"
        return record

    return make_record


def test_extra_wins_without_raising(captured):
    logger, records = captured
    with log_context(stage="training"):
        logger.info("fetch", extra={"stage": "download"})
        logger.info("fit")

    assert [r.stage for r in records] == ["download", "training"]


def test_other_key_from_extra_still_gets_context(captured):
    logger, records = captured
    with log_context(stage="training", run="exp_001"):
        logger.info("fetch", extra={"stage": "download"})

    assert (records[0].stage, records[0].run) == ("download", "exp_001")


def test_another_thread_does_not_get_the_context(captured):
    logger, records = captured
    entered, logged = threading.Event(), threading.Event()

    def other():
        entered.wait()
        logger.info("from the other thread")
        logged.set()

    thread = threading.Thread(target=other)
    thread.start()
    with log_context(run="exp_001"):
        entered.set()
        logged.wait()
        logger.info("from this thread")
    thread.join()

    by_message = {r.getMessage(): r for r in records}
    assert not hasattr(by_message["from the other thread"], "run")
    assert by_message["from this thread"].run == "exp_001"


def test_asyncio_tasks_have_their_own_context(captured):
    logger, records = captured

    async def task(run):
        with log_context(run=run):
            await asyncio.sleep(0)  # let the other task enter its context
            logger.info(run)

    async def main():
        await asyncio.gather(task("a"), task("b"))

    asyncio.run(main())
    assert {r.getMessage(): r.run for r in records} == {"a": "a", "b": "b"}


@pytest.mark.parametrize("loadenv_first", [True, False])
def test_precedence_extra_over_context_over_loadenv(captured, monkeypatch, loadenv_first):
    logger, records = captured
    base = inspect.unwrap(logging.Logger.makeRecord)
    if loadenv_first:
        # loadEnv's sitecustomize patches at startup, before CREST is imported.
        monkeypatch.setattr(logging.Logger, "makeRecord", _loadenv_style(base))
        _install_context_stamping()
    else:
        monkeypatch.setattr(logging.Logger, "makeRecord", base)
        _install_context_stamping()
        monkeypatch.setattr(logging.Logger, "makeRecord",
                            _loadenv_style(logging.Logger.makeRecord))

    logger.info("outside")
    with log_context(stage="training"):
        logger.info("inside")
        logger.info("explicit", extra={"stage": "download"})

    assert [r.stage for r in records] == ["ingest", "training", "download"]


def test_installing_twice_does_not_double_wrap(monkeypatch):
    monkeypatch.setattr(logging.Logger, "makeRecord",
                        inspect.unwrap(logging.Logger.makeRecord))
    _install_context_stamping()
    installed = logging.Logger.makeRecord
    _install_context_stamping()

    assert logging.Logger.makeRecord is installed
    assert getattr(installed, logging_service._CONTEXT_WRAPPER_ATTR)


def test_record_factory_is_left_alone():
    factory = logging.getLogRecordFactory()
    with log_context(run="exp_001"):
        assert logging.getLogRecordFactory() is factory
