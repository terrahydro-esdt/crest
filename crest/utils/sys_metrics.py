from __future__ import annotations
import os
import time
import logging
from pathlib import Path
from contextlib import contextmanager
from time import perf_counter_ns
import psutil

logger = logging.getLogger(__name__)

def _default_run_dir(base: str = "logs") -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return str(Path(base) / ts)

def emit_metric(kind: str, **fields):
    logging.getLogger("crest.metrics").info(
        kind,
        extra={"kind": kind, "ts": time.time(), **fields}
    )

class SysMetrics:
    def __init__(self, run_id: str | None, run_dir: str | None):
        run_dir = run_dir or _default_run_dir()
        self.run_id  = run_id  or Path(run_dir).name
        self.run_dir = run_dir
        self._proc = psutil.Process(os.getpid())

    def emit(self, kind: str, **fields):
        emit_metric(kind, run_id=self.run_id, run_dir=self.run_dir, **fields)

    @contextmanager
    def timed(self, label: str | None, **extra_fields):

        start_time = perf_counter_ns() / 1e9
        start_mem = self._proc.memory_info()
        start_cpu = self._proc.cpu_times()

        try: 
            yield
        finally:
            elapsed = (perf_counter_ns() / 1e9) - start_time
            end_mem = self._proc.memory_info()
            end_cpu = self._proc.cpu_times()

            self.emit(
                "timed",
                label=label,
                elapsed=elapsed,
                rss_mb=end_mem.rss / 1e6,
                rss_delta_mb=(end_mem.rss - start_mem.rss) / 1e6,
                cpu_user_s=end_cpu.user - start_cpu.user,
                cpu_sys_s=end_cpu.system - start_cpu.system,
                **extra_fields
            )

    def counter(self, label: str, value: int | float = 1, **extra_fields):
        self.emit("counter", label=label, value=value, **extra_fields)

    def gauge(self, label: str, value: int | float, **extra_fields):
        self.emit("gauge", label=label, value=value, **extra_fields)

    def snapshot(self, **fields):
        mem = self._proc.memory_info()
        cpu = self._proc.cpu_times()

        self.emit(
            "snapshot",
            rss_mb=mem.rss / 1e6,
            vms_mb=mem.vms / 1e6,
            threads=self._proc.num_threads(),
            cpu_user_s=cpu.user,
            cpu_sys_s=cpu.system,
            **fields
        )