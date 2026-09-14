"""
Part D — The cost of attention.

Naive path materializes the full (B,H,S,S) attention matrix by hand.
Fused path uses torch.nn.functional.scaled_dot_product_attention, which
dispatches to FlashAttention/memory-efficient kernels on supported cards.

Sweep: coarse pass at S in {512,1024,2048,4096,8192,16384}, catching OOM.
Then refine_oom_boundary() does a doubling-down binary search between the
largest S that succeeded and the smallest that failed, per the assignment's
requirement to report both bounds rather than assert a single exact one.

Usage: python part_d_attention.py --card rtx4090 --batch 1 --heads 8 --head-dim 64
"""
import argparse
import math
import torch
import torch.nn.functional as F

from utils import cuda_timer, warmup, log_run, peak_memory_bytes

SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 20480, 24576, 28672, 32768]


def make_qkv(batch, heads, seq_len, head_dim, dtype, device="cuda"):
    shape = (batch, heads, seq_len, head_dim)
    q = torch.randn(shape, dtype=dtype, device=device)
    k = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(shape, dtype=dtype, device=device)
    return q, k, v


def naive_attention(q, k, v):
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)  # (B,H,S,S) materialized
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def fused_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)  # backend picks flash/mem-efficient/math


def try_run(fn, q, k, v, reps=10):
    """Returns (result_dict, oom: bool). Resets peak memory before each try."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        warmup(lambda: fn(q, k, v), n=3)
        with cuda_timer(repetitions=reps) as t:
            for _ in range(reps):
                fn(q, k, v)
        peak_bytes = torch.cuda.max_memory_allocated()
        return {"seconds_per_call": t["seconds"],
                "peak_memory_bytes": peak_bytes, "oom": False}, False
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"oom": True}, True


def sweep(fn, label, batch, heads, head_dim, dtype, card, part_tag, reps=10):
    results = []
    for seq_len in SEQ_LENGTHS:
        q, k, v = make_qkv(batch, heads, seq_len, head_dim, dtype)
        result, oom = try_run(fn, q, k, v, reps=reps)
        result.update({"impl": label, "seq_len": seq_len, "batch": batch,
                        "heads": heads, "head_dim": head_dim})
        log_run(part_tag, config={"impl": label, "seq_len": seq_len,
                                   "batch": batch, "heads": heads,
                                   "head_dim": head_dim, "reps": reps},
                 result=result)
        if oom:
            print(f"[{label}] S={seq_len:>6}: OOM")
        else:
            ms = result["seconds_per_call"] * 1000
            gb = result["peak_memory_bytes"] / 1e9
            print(f"[{label}] S={seq_len:>6}: {ms:.2f} ms, {gb:.2f} GB peak")
        results.append(result)
        del q, k, v
        torch.cuda.empty_cache()
        if oom:
            break
    return results


def refine_oom_boundary(fn, batch, heads, head_dim, dtype, last_ok, first_oom,
                         reps=5, resolution=1):
    """Binary search between the largest S that ran and the smallest that
    OOM'd, down to `resolution` tokens. Returns (largest_ok, smallest_oom)."""
    lo, hi = last_ok, first_oom
    while hi - lo > resolution:
        mid = (lo + hi) // 2
        q, k, v = make_qkv(batch, heads, mid, head_dim, dtype)
        _, oom = try_run(fn, q, k, v, reps=reps)
        del q, k, v
        torch.cuda.empty_cache()
        if oom:
            hi = mid
        else:
            lo = mid
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--dtype", default="float16",
                     choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--mem-fraction", type=float, default=1.0,
                     help="Cap the CUDA allocator to this fraction of total "
                          "VRAM (0.0-1.0). On Windows/WDDM, uncapped "
                          "allocations can silently spill into system RAM "
                          "instead of raising OutOfMemoryError, which hides "
                          "the true OOM boundary the assignment asks for. "
                          "Set e.g. 0.9 to force a real CUDA OOM within "
                          "physical VRAM. State whatever value you use in "
                          "METRICS.md.")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    assert torch.cuda.is_available(), "CUDA not available — run this on the GPU box"
    if args.mem_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
        print(f"Capped CUDA allocator to {args.mem_fraction:.0%} of VRAM "
              f"to force a genuine OOM (avoids WDDM system-RAM spillover)")
    print(f"Config: batch={args.batch} heads={args.heads} "
          f"head_dim={args.head_dim} dtype={args.dtype} (state these in METRICS.md)")

    print("\n=== Naive attention ===")
    naive_results = sweep(naive_attention, "naive", args.batch, args.heads,
                           args.head_dim, dtype, args.card, "D", args.reps)

    print("\n=== Fused attention (SDPA) ===")
    fused_results = sweep(fused_attention, "fused", args.batch, args.heads,
                           args.head_dim, dtype, args.card, "D", args.reps)

    for label, results, fn in [("naive", naive_results, naive_attention),
                                ("fused", fused_results, fused_attention)]:
        ok = [r["seq_len"] for r in results if not r["oom"]]
        oom = [r["seq_len"] for r in results if r["oom"]]
        if ok and oom:
            last_ok, first_oom = max(ok), min(oom)
            lo, hi = refine_oom_boundary(fn, args.batch, args.heads,
                                          args.head_dim, dtype, last_ok, first_oom)
            print(f"{label}: largest OK={lo}, smallest OOM={hi} "
                  f"(refined from coarse sweep step {last_ok}->{first_oom})")
            log_run("D", config={"impl": label, "refine": True},
                     result={"largest_ok": lo, "smallest_oom": hi})
        elif ok:
            print(f"{label}: no OOM observed up to S={max(ok)} — "
                  f"extend SEQ_LENGTHS if you need the actual boundary")


if __name__ == "__main__":
    main()