""" 
This is a general purpose module to measure
the performance of code incluidng memory usage
and time.
"""

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
    """ Sets the default path for logfile """
    ts = time.strftime("%Y%m%d-%H%M%S")
    return str(Path(base) / ts)

class SysMetrics:
    """
    SysMetrics measures the memory usage and performance (time)
    of any code. 

    Parameters
    ----------
    run_id : str | None
        Optional id/name to associate with a particular 
        wrapped code section being measured.
    
    run_dir : str | None
        Optional path where metric logs are written.
    
    Examples
    --------
    >>> sm = SysMetrics()
    >>> with sm.timed("function"):
    >>>     # code to be measured

    """
    def __init__(self, run_id: str | None, run_dir: str | None):
        run_dir = run_dir or str(Path("logs") / time.strftime("%Y%m%d-%H%M%S"))
        self.run_id  = run_id  or Path(run_dir).name
        self.run_dir = run_dir
        self._proc = psutil.Process(os.getpid())

    def emit(self, kind: str, **fields):
        """ Writes the metrics to log """
        logging.getLogger("crest.metrics").info(
            kind,
            extra={"kind": kind, "ts": time.time(), **fields}
        )

        emit_metric(kind, run_id=self.run_id, run_dir=self.run_dir, **fields)

    @contextmanager
    def timed(self, label: str | None, **extra_fields):
        """ Times the wrapped code """

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

    def snapshot(self, **fields):
        """ Snapshot of current system usage: memory usage, # threads, ...  """
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
