"""
Part C — Bandwidth-bound vs compute-bound.

Memory-bound op: large elementwise copy (out = in.clone()), touches
2*N*4 bytes (read+write) per call for FP32.
Compute-bound op: large square FP32 matmul, reusing Part B's flop count.

Arithmetic intensity (FLOPs/byte) is computed for each and compared
against the card's roofline ridge point (peak_TFLOPS / peak_bandwidth_GBs),
which you fill in once you have the Part A vendor numbers.

Usage: python part_c_bandwidth.py --card rtx4090 --elems 268435456
       (268435456 floats = 1 GiB per buffer; adjust to fit VRAM)
"""
import argparse
import torch

from utils import cuda_timer, warmup, log_run

BYTES_PER_FLOAT32 = 4


def bench_memory_bound(n_elems: int, reps: int, device="cuda"):
    src = torch.randn(n_elems, dtype=torch.float32, device=device)
    dst = torch.empty_like(src)

    def op():
        dst.copy_(src)

    warmup(op, n=10)
    with cuda_timer(repetitions=reps) as t:
        for _ in range(reps):
            op()
    seconds = t["seconds"]

    bytes_moved = 2 * n_elems * BYTES_PER_FLOAT32  # 1 read + 1 write
    effective_gbs = (bytes_moved / seconds) / 1e9
    # elementwise copy does 0 FLOPs, so arithmetic intensity is 0 — it's the
    # textbook memory-bound extreme, worth stating explicitly in METRICS.md
    return {
        "op": "elementwise_copy",
        "n_elems": n_elems,
        "seconds_per_call": seconds,
        "bytes_moved": bytes_moved,
        "effective_gbs": effective_gbs,
        "flops": 0,
        "arithmetic_intensity": 0.0,
    }


def bench_compute_bound(n: int, reps: int, device="cuda"):
    a = torch.randn(n, n, dtype=torch.float32, device=device)
    b = torch.randn(n, n, dtype=torch.float32, device=device)

    warmup(lambda: torch.matmul(a, b), n=10)
    with cuda_timer(repetitions=reps) as t:
        for _ in range(reps):
            torch.matmul(a, b)
    seconds = t["seconds"]

    flops = 2.0 * (n ** 3)
    # bytes touched: read A, read B, write C once each (ignores cache reuse,
    # which is exactly why matmul's *effective* AI is much higher than this
    # naive bound — note that distinction in your writeup)
    bytes_touched = 3 * (n * n) * BYTES_PER_FLOAT32
    arithmetic_intensity = flops / bytes_touched
    return {
        "op": "square_matmul",
        "n": n,
        "seconds_per_call": seconds,
        "flops": flops,
        "achieved_tflops": flops / seconds / 1e12,
        "bytes_touched": bytes_touched,
        "arithmetic_intensity": arithmetic_intensity,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--elems", type=int, default=1 << 28,
                     help="elements per buffer for the copy kernel")
    ap.add_argument("--matmul-n", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available — run this on the GPU box"

    mem_result = bench_memory_bound(args.elems, args.reps)
    log_run("C", config={"kind": "memory_bound", "elems": args.elems,
                          "reps": args.reps}, result=mem_result)
    print("Memory-bound (copy):", mem_result)

    compute_result = bench_compute_bound(args.matmul_n, args.reps)
    log_run("C", config={"kind": "compute_bound", "n": args.matmul_n,
                          "reps": args.reps}, result=compute_result)
    print("Compute-bound (matmul):", compute_result)

    print(f"\nArithmetic intensity — copy: {mem_result['arithmetic_intensity']:.4f} "
          f"FLOPs/byte, matmul(N={args.matmul_n}): "
          f"{compute_result['arithmetic_intensity']:.2f} FLOPs/byte")
    print("Compare each against the card's roofline ridge point "
          "(peak_TFLOPS*1e12 / peak_bandwidth_GBs*1e9) to state which side "
          "of the roofline each operation falls on.")


if __name__ == "__main__":
    main()
