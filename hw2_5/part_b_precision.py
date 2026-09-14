"""
Part B — Precision and achieved throughput.

Benchmarks dense NxN @ NxN matmul at N in {1024,4096,8192,16384} across
FP32, TF32, FP16, BF16. Writes one RUN_LOG.txt line per (N, precision) and
a CSV for plotting.

Peak TFLOPS table below is what you need to fill in from vendor docs
(Part A) to get "% of theoretical peak" — left blank/None here on purpose,
computed in analyze_and_plot.py once you've filled THEORETICAL_PEAK_TFLOPS.

Usage: python part_b_precision.py --card rtx4090 --reps 30
"""
import argparse
import torch

from utils import cuda_timer, warmup, log_run, append_csv

SIZES = [1024, 4096, 8192, 16384]

# precision -> (torch dtype for the operands, whether to force TF32 matmul)
PRECISIONS = {
    "FP32": (torch.float32, False),
    "TF32": (torch.float32, True),   # TF32 is a compute mode, not a storage dtype
    "FP16": (torch.float16, False),
    "BF16": (torch.bfloat16, False),
}


def matmul_flops(n: int) -> float:
    # 2*N^3 FLOPs for an NxN @ NxN matmul (multiply+add per output element sum)
    return 2.0 * (n ** 3)


def run_one(n: int, precision: str, reps: int, device="cuda"):
    dtype, use_tf32 = PRECISIONS[precision]
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

    a = torch.randn(n, n, dtype=dtype, device=device)
    b = torch.randn(n, n, dtype=dtype, device=device)

    warmup(lambda: torch.matmul(a, b), n=10)

    with cuda_timer(repetitions=reps) as t:
        for _ in range(reps):
            torch.matmul(a, b)
    seconds_per_call = t["seconds"]

    achieved_tflops = matmul_flops(n) / seconds_per_call / 1e12
    return {
        "n": n,
        "precision": precision,
        "reps": reps,
        "seconds_per_call": seconds_per_call,
        "achieved_tflops": achieved_tflops,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True, help="label, e.g. rtx4090 or rtx5090")
    ap.add_argument("--reps", type=int, default=30,
                     help="repetitions per config; report this in METRICS.md")
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()
    csv_path = args.csv_out or f"{args.card}_part_b_throughput.csv"

    assert torch.cuda.is_available(), "CUDA not available — run this on the GPU box"

    for n in SIZES:
        for precision in PRECISIONS:
            try:
                result = run_one(n, precision, args.reps)
            except torch.cuda.OutOfMemoryError:
                result = {"n": n, "precision": precision, "reps": args.reps,
                           "seconds_per_call": None, "achieved_tflops": None,
                           "oom": True}
                torch.cuda.empty_cache()

            log_run("B", config={"n": n, "precision": precision, "reps": args.reps},
                     result=result)
            append_csv(csv_path, {"card": args.card, **result})
            print(f"N={n:>6} {precision:>5}: "
                  f"{result.get('achieved_tflops', 'OOM')} TFLOPS")


if __name__ == "__main__":
    main()
