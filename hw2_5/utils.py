"""
Shared utilities for HW2.5. Every measurement written via log_run() is
tagged with the GPU UUID so results in METRICS.md are traceable back to
RUN_LOG.txt as the assignment requires.
"""
import csv
import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import torch

RUN_LOG_PATH = os.environ.get("RUN_LOG_PATH", "RUN_LOG.txt")


def get_gpu_uuid(device_index: int = 0) -> str:
    """Query nvidia-smi directly rather than trusting torch, since torch
    gives you a name/index but not the UUID we need for traceability."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader",
             f"-i={device_index}"],
            text=True,
        )
        return out.strip()
    except Exception as e:
        return f"UNKNOWN-nvidia-smi-failed:{e}"


def get_gpu_name(device_index: int = 0) -> str:
    return torch.cuda.get_device_name(device_index)


def log_run(part: str, config: dict, result: dict, device_index: int = 0):
    """Append one structured line to RUN_LOG.txt. Do this for every single
    measurement — Table HW2.5.1 must trace every cell back to a line here."""
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "part": part,
        "gpu_uuid": get_gpu_uuid(device_index),
        "gpu_name": get_gpu_name(device_index),
        "config": config,
        "result": result,
    }
    with open(RUN_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def append_csv(path: str, row: dict):
    """Append a row to a CSV, writing the header only if the file is new.
    Used for the thermal log (Part E) and any sweep you want in a plotting-
    friendly format alongside the JSONL RUN_LOG."""
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@contextmanager
def cuda_timer(repetitions: int = 1):
    """Context manager returning elapsed seconds via CUDA events, which are
    accurate for GPU-side work (unlike time.perf_counter, which would
    include Python dispatch overhead and unsynchronized kernel launches).
    Caller is responsible for warmup before entering this block."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timer_box = {"seconds": None}
    start.record()
    yield timer_box
    end.record()
    torch.cuda.synchronize()
    timer_box["seconds"] = start.elapsed_time(end) / 1000.0 / repetitions


def warmup(fn, n: int = 10):
    """Run fn() n times and synchronize before any timed region. Required
    because the first few kernel launches pay JIT/cache/clock-ramp costs
    that would otherwise pollute your throughput numbers."""
    for _ in range(n):
        fn()
    torch.cuda.synchronize()


def peak_memory_bytes(device_index: int = 0) -> int:
    torch.cuda.reset_peak_memory_stats(device_index)
    return torch.cuda.max_memory_allocated(device_index)
