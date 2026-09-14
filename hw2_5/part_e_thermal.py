"""
Part E — Sustained load and thermal behaviour.

Runs a continuous compute-bound matmul loop on a background thread while
the main thread polls nvidia-smi every 5 seconds for clock/temp/power/util
and appends to a CSV. Splitting compute and sampling into two threads
avoids the sampling calls themselves stalling the GPU queue.

Usage: python part_e_thermal.py --card rtx4090 --minutes 20 --matmul-n 8192
"""
import argparse
import csv
import subprocess
import threading
import time

import torch

from utils import get_gpu_uuid

SAMPLE_INTERVAL_S = 5

SMI_QUERY_FIELDS = [
    "clocks.sm", "clocks.mem", "temperature.gpu",
    "power.draw", "utilization.gpu",
]


def sample_nvidia_smi(device_index: int = 0) -> dict:
    query = ",".join(SMI_QUERY_FIELDS)
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits",
         f"-i={device_index}"],
        text=True,
    ).strip()
    values = [v.strip() for v in out.split(",")]
    return dict(zip(SMI_QUERY_FIELDS, values))


class ComputeLoad:
    """Runs matmul in a tight loop on a background thread, tracking
    completed-call timestamps so we can compute throughput in the first
    30s vs the final 5 minutes afterward."""

    def __init__(self, n: int, dtype=torch.float32, device="cuda"):
        self.n = n
        self.dtype = dtype
        self.device = device
        self._stop = threading.Event()
        self.call_timestamps = []
        self._lock = threading.Lock()

    def _worker(self):
        a = torch.randn(self.n, self.n, dtype=self.dtype, device=self.device)
        b = torch.randn(self.n, self.n, dtype=self.dtype, device=self.device)
        while not self._stop.is_set():
            torch.matmul(a, b)
            torch.cuda.synchronize()
            with self._lock:
                self.call_timestamps.append(time.monotonic())

    def start(self):
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._t0 = time.monotonic()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=30)

    def throughput_in_window(self, start_s: float, end_s: float) -> float:
        """Calls completed per second within [start_s, end_s] of run time."""
        with self._lock:
            rel = [ts - self._t0 for ts in self.call_timestamps]
        n_calls = sum(1 for r in rel if start_s <= r < end_s)
        window = end_s - start_s
        return n_calls / window if window > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--minutes", type=int, default=20)
    ap.add_argument("--matmul-n", type=int, default=8192)
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()
    csv_path = args.csv_out or f"{args.card}_thermal_log.csv"
    total_seconds = args.minutes * 60

    assert torch.cuda.is_available(), "CUDA not available — run this on the GPU box"
    uuid = get_gpu_uuid()

    load = ComputeLoad(n=args.matmul_n)
    load.start()

    fieldnames = ["elapsed_s", "gpu_uuid"] + SMI_QUERY_FIELDS
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        t0 = time.monotonic()
        next_sample = t0
        while time.monotonic() - t0 < total_seconds:
            now = time.monotonic()
            if now >= next_sample:
                sample = sample_nvidia_smi()
                row = {"elapsed_s": round(now - t0, 1), "gpu_uuid": uuid, **sample}
                writer.writerow(row)
                f.flush()
                print(row)
                next_sample += SAMPLE_INTERVAL_S
            time.sleep(0.5)

    load.stop()

    # Peak throughput in first 30s vs steady-state in final 5 minutes.
    # NOTE: only valid if total_seconds > 330 (30s + 5min); enforced by
    # --minutes default of 20.
    first_30 = load.throughput_in_window(0, 30)
    final_5min_start = max(0, total_seconds - 300)
    steady = load.throughput_in_window(final_5min_start, total_seconds)
    pct = (steady / first_30 * 100) if first_30 > 0 else float("nan")

    print(f"\nPeak throughput (first 30s): {first_30:.3f} calls/s")
    print(f"Steady-state (final 5 min): {steady:.3f} calls/s")
    print(f"Steady-state / peak: {pct:.1f}%")
    print(f"Thermal log written to {csv_path} — plot clocks.sm/temperature.gpu "
          f"vs elapsed_s to find throttle onset.")


if __name__ == "__main__":
    main()
