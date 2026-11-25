# METRICS — DATA 266 · HW11 (repo folder HW12)

**Student:** Ritika Mukesh Neema · SID4 6638
**Step 0:** SEED=6638, SLICE=638, HP_ID=2, CLS_A=8, CLS_B=5 (HP_ID adds no temperature in HW11)

The two parts are reported separately because they use different models, tasks and GPUs.

---

## Part 1 — GPTQ (base LLM vs 4-bit GPTQ)

| Setting | Value |
|---|---|
| Base model | `facebook/opt-125m` @ revision `27dcfa74d334bc871f3234de431e71c6eeba5dd6` (125.2M params) |
| Hardware | NVIDIA L4, same GPU for both models |
| Stack | torch 2.15.0+cu132, transformers 5.24.1, gptqmodel 7.12.2, optimum 2.3.0, datasets 5.2.0 |
| GPTQ | 4-bit, group size 128, symmetric, desc_act=False; 128 × 512-token calibration windows from Wikitext-2 **train** |
| Quality eval | `Salesforce/wikitext` `wikitext-2-raw-v1` **test**, 140 non-overlapping windows × 2,048 tokens |
| Latency | batch 1, 5 warm-up + 50 timed runs; prefill = 1 forward pass of 512 tokens; generate = 128-token prompt + 64 forced new tokens (greedy) |
| Quantization time | 106.6 s |
| GPTQ inference kernel | `MarlinLinear` |

| Metric | FP16 base | GPTQ INT4 | GPTQ / base |
|---|---|---|---|
| Wikitext-2 eval loss (nats/token) | 3.3198 | 3.4293 | +0.1095 |
| Perplexity | 27.656 | 30.855 | +11.57% |
| Prefill p50 / p95 (ms) | 10.14 / 11.05 | 20.23 / 20.85 | 1.99× |
| Generate p50 / p95 (ms) | 617.9 / 641.4 | 1240.8 / 1281.4 | 2.01× / 2.00× |
| Weights on GPU (MiB) | 245.5 | 118.7 | 0.484 |
| Peak GPU memory (MiB) | 1570.8 | 1444.0 | 0.919 |
| On-disk size (MiB) | 242.3 | 123.0 | 0.508 |

---

## Part 2 — Knowledge distillation (BERT teacher → DistilBERT student, AG News)

| Setting | Value |
|---|---|
| Teacher | Instructor's fine-tuned `BertForSequenceClassification` (`teacher_model.7z`), 109.5M params |
| Student | `distilbert/distilbert-base-uncased`, 67.0M params, re-initialised from the same checkpoint for every run |
| Hardware | Tesla T4, same GPU for teacher and student |
| Stack | torch 2.14.1+cu130, transformers 5.22.1, datasets 5.2.0 |
| Data | AG News: 114,000 train / 6,000 validation (seeded 5% hold-out) / 7,600 test; teacher logits extracted for all 120,000 training samples (72 s) |
| Loss | α·CE(student, y) + (1−α)·T²·KL(softmax(z_t/T) ‖ softmax(z_s/T)) |
| Fixed training settings | 2 epochs, batch 32, AdamW lr 5e-5, weight decay 0.01, 6% warm-up then linear decay, grad clip 1.0, max length 128, fp16 autocast, SEED 6638 |
| Selection rule | best **validation** accuracy |

**Teacher sanity check:** 98.70% on 2,000 training samples; output columns already in AG News label order. Teacher accuracy on the full training set: 98.95%.

### Temperature sweep (α = 0.5)

| T | Val acc | Test acc | Final CE | Final T²·KL | Train time (min) |
|---|---|---|---|---|---|
| 1 | 0.9450 | 0.9488 | 0.1054 | 0.0756 | 11.19 |
| 2 | 0.9447 | 0.9480 | 0.1252 | 0.1997 | 11.14 |
| **4** | **0.9457** | 0.9483 | 0.1450 | 0.3110 | 11.21 |
| 8 | 0.9448 | 0.9464 | 0.1488 | 0.3355 | 11.15 |

**Best T = 4.**

### α sweep (T = 4)

| α (weight on CE) | Val acc | Test acc | Final CE | Final T²·KL | Train time (min) |
|---|---|---|---|---|---|
| **0.25** | **0.9477** | 0.9468 | 0.1566 | 0.3238 | 11.15 |
| 0.50 | 0.9457 | 0.9483 | 0.1450 | 0.3110 | 11.15 |
| 0.75 | 0.9460 | 0.9476 | 0.1280 | 0.3094 | 11.17 |

**Best α = 0.25.** Final student: T = 4, α = 0.25. The repeated T=4/α=0.5 run reproduced its first result exactly.
Noise reference: one standard error ≈ 0.29 pts on validation (n = 6,000) and ≈ 0.26 pts on test (n = 7,600).

### Teacher vs student (AG News test set, same T4, fp16 autocast)

| Metric | Teacher (BERT) | Student (DistilBERT) | Change |
|---|---|---|---|
| Test accuracy | 0.9443 | 0.9468 | +0.25 pts (100.26% retained) |
| Full test set, batch 64 (s) | 6.07 | 3.02 | 2.01× faster |
| Throughput (examples/s) | 1,251 | 2,519 | 2.01× |
| Batch-1 latency p50 / p95 (ms)* | 29.75 / 56.91 | 6.00 / 7.48 | 4.96× (p50) |
| Peak GPU memory (MiB) | 796.5 | 527.3 | −33.8% |
| Parameters (M) | 109.5 | 67.0 | −38.8% |
| On-disk size (MiB, fp32 safetensors) | 417.7 | 255.4 | −38.8% |

\*The teacher's batch-1 latency was measured without a batch-1 warm-up and is probably inflated; the batched 2.0× figure is the reliable speedup.

### Per-class test accuracy

| Class | Teacher | Student | Drop (pts) |
|---|---|---|---|
| World | 0.9489 | 0.9526 | −0.37 |
| Sports | 0.9884 | 0.9889 | −0.05 |
| **Business** | **0.9137** | **0.9111** | **+0.26** |
| Sci/Tech | 0.9263 | 0.9347 | −0.84 |

Most degraded class: **Business**. 45 Business test examples are right for the teacher and wrong for the student (student predicted Sci/Tech 30, World 13, Sports 2). Examples used in the analysis: test indices 2281, 2914, 6980, 7413, 7470.

---

## Cross-part summary (relative changes only; different tasks and GPUs)

| Relative to the uncompressed model | GPTQ INT4 (Part 1) | Distillation (Part 2) |
|---|---|---|
| On-disk size | 0.51× | 0.61× |
| Peak GPU memory | 0.92× (weights 0.48×) | 0.66× |
| Speed | 2.0× slower (generation p50) | 2.0× faster (throughput) |
| Quality, in its own metric | perplexity +11.6% | accuracy +0.25 pts |
| Cost to produce | 106.6 s calibration | 7 runs × ~11.2 min |
