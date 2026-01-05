from __future__ import annotations
import os, time, logging
from dataclasses import dataclass, field
from typing import Callable, Optional
from pathlib import Path
import psutil, os
from time import perf_counter_ns
import socket
from contextlib import contextmanager

from .Stopwatch import Stopwatch

logger = logging.getLogger(__name__)

def now_s(): 
    return perf_counter_ns()/1e9

def _default_run_dir(base: str = "logs") -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    host = socket.gethostname()
    run_dir = Path(base) / f"{ts}_{host}"
    return str(run_dir)

def emit_metric(kind: str, **fields):
    stage = fields.get("stage") or fields.get("desc") or ""
    msg = f"{kind}:{stage}" if stage else kind
    logging.getLogger("crest.metrics").info(msg, extra={"kind": kind, "ts": time.time(), **fields})

@contextmanager
def emitting_stopwatch(message, emit, **sw_kw):
    sw = Stopwatch(message=message, **sw_kw)
    with sw:
        yield
    try:
        emit(sw.deltas | {"desc": message})
    except Exception:
        pass

@dataclass
class RunContext:
    run_id: str
    run_dir: str
    dataset_info: dict = field(default_factory=dict)
    model_info: dict   = field(default_factory=dict)
    totals: dict       = field(default_factory=lambda: {
        "records": 0, "batches": 0, "bytes_in": 0, "bytes_out": 0
    })

class SysMetrics:
    """Collects structured metrics (JSONL) for stages & batches using Stopwatch."""
    def __init__(self, run_id: Optional[str]=None, run_dir: Optional[str]=None):
        run_dir = run_dir or _default_run_dir()
        run_id  = run_id  or Path(run_dir).name
        self.ctx = RunContext(run_id=run_id, run_dir=run_dir)

    def _emit(self, kind: str, **fields):
        base = {
            "run_id": self.ctx.run_id,
            "run_dir": self.ctx.run_dir,
            **fields
        }
        emit_metric(kind, **base)


    def _stopwatch(self, message: str, name: Optional[str]=None,
                extra_metrics: dict[str, Callable]=None,
                silent: bool|dict=False, stop_gc=False):

        silent = {} if silent is False else silent

        def jsonl_emit(deltas: dict):
            self._emit("stage", stage=name or message, desc=message, deltas=deltas)

        return emitting_stopwatch(
            message=message,
            emit=jsonl_emit,
            logger=logger.info,
            timer=(lambda: perf_counter_ns()/1e9),
            metrics=extra_metrics or {},
            silent=silent,
            stop_gc=stop_gc,
        )
    
    @staticmethod
    def psutil_metrics():
        proc = psutil.Process(os.getpid())

        def rss_mb(): return proc.memory_info().rss / 1e6
        def vms_mb(): return proc.memory_info().vms / 1e6
        def threads(): return proc.num_threads()
        def cpu_user_s(): return proc.cpu_times().user
        def cpu_sys_s():  return proc.cpu_times().system

        def io_read_b():
            try: return proc.io_counters().read_bytes
            except Exception: return None

        def io_write_b():
            try: return proc.io_counters().write_bytes
            except Exception: return None

        return {
            "rss_mb": rss_mb,
            "vms_mb": vms_mb,
            "threads": threads,
            "cpu_user_s": cpu_user_s,
            "cpu_sys_s": cpu_sys_s,
            "io_read_b": io_read_b,
            "io_write_b": io_write_b,
        }

    def record_dataset_info(self, *, start_dt=None, end_dt=None,
                            lat_range=None, lon_range=None,
                            n_rows: Optional[int]=None,
                            region:str|None=None, extent:str|None=None,
                            time_steps: Optional[int]=None,
                            n_points: Optional[int]=None,
                            n_workers: Optional[int]=None):
        info = {
            "lat_range": lat_range, "lon_range": lon_range,
            "n_rows": n_rows, "region": region, "extent": extent,
            "time_steps": time_steps, "points": n_points, "workers": n_workers
        }
        if start_dt is not None or end_dt is not None:
            info["datetime_range"] = [str(start_dt) if start_dt is not None else None,
                                    str(end_dt)   if end_dt   is not None else None]
        self.ctx.dataset_info.update({k: v for k, v in info.items() if v is not None})
        self._emit("dataset_info", **self.ctx.dataset_info)

    def record_model_info(self, *, name: str, path: str, version: Optional[str]=None):
        self.ctx.model_info.update({"model_name": name, "path": path, "version": version})
        self._emit("model_info", **self.ctx.model_info)

    def stage(self, name: str, stop_gc: bool=False):
        return self._stopwatch(message=name, name=name, stop_gc=stop_gc)

    def record_batch(self, batch_idx:int, size:int|None=None):
        def do_emit(deltas):
            self.ctx.totals["batches"] += 1
            if size is not None: 
                self.ctx.totals["records"] += int(size)

            self._emit("batch", batch_idx=batch_idx, size=size, deltas=deltas)

        return emitting_stopwatch(
            message=f"batch[{batch_idx}]",
            emit=do_emit,
            logger=logging.getLogger(__name__).info,
            timer=(lambda: perf_counter_ns()/1e9),
            silent=True,
        )

    def record_bytes(self, *, in_bytes: Optional[int]=None, out_bytes: Optional[int]=None):
        if in_bytes is not None:
            self.ctx.totals["bytes_in"]  += int(in_bytes)
        if out_bytes is not None:
            self.ctx.totals["bytes_out"] += int(out_bytes)
        bi, bo = self.ctx.totals["bytes_in"], self.ctx.totals["bytes_out"]
        self._emit("io", bytes_in=bi, bytes_out=bo)

    def finalize(self, ok: bool=True):
        self._emit("run_summary",
                   ok=ok, **self.ctx.totals,
                   dataset_info=self.ctx.dataset_info,
                   model_info=self.ctx.model_info)
        return self.ctx