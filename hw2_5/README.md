Student: Ritika Mukesh Neema
SJSU ID: 019306638
SJSU ID Seed: 6638

# HW2.5 GPU Benchmark Suite

Verified with `py_compile` on all files and unit-checked math/logic on a
CPU-only sandbox (FLOPs formulas, arithmetic-intensity algebra, OOM
binary-search convergence, throughput-window aggregation, quadratic-fit
recovery — see inline comments). No CUDA-dependent code has run yet; run
each step below on the RTX 5090 / RTX 4090 box and sanity-check the printed
numbers before moving to the next part.

## Setup
```
pip install -r requirements.txt
```

## Run order (repeat entire sequence per card)
```
python part_a_provenance.py --out-prefix rtx4090
# -> fill vendor-doc fields (architecture, mem type/bandwidth, tensor core
#    gen, supported precisions) into a provenance_manual.json yourself, with
#    a source URL. Not automatable — that's a citation requirement.

python part_b_precision.py --card rtx4090 --reps 30
python part_c_bandwidth.py --card rtx4090
python part_d_attention.py --card rtx4090 --batch 1 --heads 8 --head-dim 64
python part_e_thermal.py --card rtx4090 --minutes 20 --matmul-n 8192

python make_plots.py --card rtx4090
```

Every measurement appends a UUID-tagged JSON line to `RUN_LOG.txt`
(shared across cards — don't delete between runs). CSVs and PNGs are
per-card, prefixed by `--card`.

## Verify after each part before moving on
- **Part A**: open `<card>_provenance_auto.json`, confirm UUID/driver/CUDA
  match what you expect for that machine.
- **Part B**: sanity-check the printed TFLOPS aren't absurd (BF16 on a 4090
  should land well above FP32; if BF16 < FP32, something's wrong with the
  dtype dispatch — check `torch.backends.cuda.matmul.allow_tf32` isn't stuck
  on).
- **Part C**: effective GB/s from the copy kernel should be within ~80-95%
  of the card's spec bandwidth; if it's far lower, increase `--elems` so the
  copy runs long enough to amortize launch overhead.
- **Part D**: confirm `fused` survives to a much larger sequence length than
  `naive` before OOM — if they OOM at the same point, SDPA likely fell back
  to the naive math backend (check `torch.backends.cuda.*_sdp_enabled()`).
- **Part E**: watch the live printed samples for the first minute; if
  `clocks.sm` is already dropping in that window, your card was likely
  still warm from a prior part — let it idle first.

## Things still requiring manual work
- Vendor spec citations (Part A)
- METRICS.md narrative + Table HW2.5.1 (numbers come from RUN_LOG.txt /
  CSVs, but the prose — plateau explanations, roofline classification,
  fused-kernel explanation — is yours to write)
- AI_USE.md
- Reservation and GPU-hour records
