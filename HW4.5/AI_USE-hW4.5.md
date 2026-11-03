# AI_USE.md - HW4.5 (DRAFT: edit every section into your own words before submitting)

## Tools used
Claude (Anthropic) via Claude Code/claude.ai: wrote and debugged the HW4.5 scripts (train_bert.py, batch_scaling.py, part_b_full.py, vllm_bench.py, kv_cache.py, kv_measure.py), generated the Colab notebooks, and drafted METRICS.md from my notebook outputs.
## What I did myself
 chose the GPUs/Colab runtimes, ran every notebook, supplied the course SEED/SLICE convention, read the numbers, checked METRICS.md against the outputs, wrote the final analysis/conclusions.

## What went wrong and how I found it
1. Wrong seed and rows. The first runs used SEED 42 and rows 0-4999 instead of the course convention (SEED 6638, SLICE 638). I found it when comparing against the course convention and the handout clarification, then reran on the A100 and T4 with the correct values; the earlier numbers were discarded.
2. Memory limit misread. My first KV-cache probe reported 32,768 tokens as the longest context, which looked like a memory limit but is the model's own position limit. I found it because the engine's KV log showed room for about 10x more tokens. The probe now records the failure reason; the later low-memory probe (gpu_util 0.08) failed because the engine could not start, so I reported the engine's KV capacity instead.
3. Accuracy at large batch. Max-batch runs scored 0.58-0.62 and looked like technique damage. Counting optimizer steps (39 vs 654) showed it was undertraining; same-step comparisons (accumulation vs plain batch 64) showed no real technique effect.
4. A claim that runs were non-deterministic was wrong. The baseline reproduced exactly on the same GPU; different GPUs give different curves. I removed the claim.
5. Tooling: argparse rejected `--extra "--amp"`; fixed as `--extra=--amp`.

## How I verified the AI's work
 compared METRICS.md tables with the executed notebook outputs; checked KV bytes against the formula (12,288 B/token, measured = formula); recomputed GPU-hours (10,464 / 43.4 / 3600 = 0.067 h).


