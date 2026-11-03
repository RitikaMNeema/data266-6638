# HW4.5 METRICS - GPU Assignment II: Training Throughput and Memory Engineering

## 0. Setup (both cards)
| | Card 1: Tesla T4 (full measurements) | Card 2: A100-SXM4-40GB (HW4.a baseline, Parts A, B at batch 16, D) |
|---|---|---|
| `nvidia-smi -L` | GPU 0: Tesla T4 (UUID: GPU-45a3688f-91da-aece-1fc4-c05b615f9aa2) | GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-a7917fc8-eb36-56a0-5ead-9ce94cf71b01) |
| VRAM | 15,360 MiB | 40,960 MiB |
| Driver / torch / transformers | 580.82.07 / 2.11.0+cu130 / 5.22.0 (checked in a later Colab A100 session; same as the T4 run) (`nvidia-smi`, `torch.__version__`) |
| Platform | Google Colab (no reservation needed) | Google Colab (no reservation needed) |

**Data (identical for HW4.a and HW4.5):** `cardiffnlp/tweet_eval`, config `sentiment`, split `train`, **rows 638-5637** (5,000 consecutive rows from SLICE = 638), split 70/15/15 = 3,500 / 750 / 750 with **SEED = 6638** (also the training seed). No rows picked or reshuffled for HW4.5.
**Fine-tune (unchanged from HW4.a):** bert-base-uncased, max_len 128, AdamW lr 2e-5, weight decay 0.01, batch 16, 3 epochs, fp32.
**Part C:** Part C skipped (optional).
**Cross-card note:** the same seed gives slightly different training curves on the two GPUs (different kernels and precision paths), so accuracy numbers are compared only within a card.

## 1. HW4.a - Fine-tune baseline
Epoch curves (batch 16):
| Epoch | Train loss (A100) | Val loss (A100) | Val acc (A100) | Train loss (T4) | Val loss (T4) | Val acc (T4) |
|---|---|---|---|---|---|---|
| 1 | 0.858 | 0.742 | 0.669 | 0.856 | 0.732 | 0.652 |
| 2 | 0.582 | 0.743 | **0.684** | 0.564 | 0.781 | **0.664** |
| 3 | 0.330 | 0.917 | 0.648 | 0.331 | 0.896 | 0.660 |

| | A100 | T4 |
|---|---|---|
| Peak GPU memory | 2,537 MB | 2,537 MB |
| Training time (3 epochs) | 55.0 s (wall-clock 58.2 s) | 240.9 s |
| Test accuracy (epoch-3 model) | 0.700 | 0.713 |

**Where validation stopped improving:** epoch 2 (validation accuracy peaks at 0.684 on the A100 and 0.664 on the T4; validation loss is already flat or rising from epoch 1-2). I trained one more epoch. Cost: on the A100 about 18 s of training, with val accuracy down 3.6 points (0.684 to 0.648) and val loss up from 0.743 to 0.917; on the T4 about 83 s (a third of the run), val loss up from 0.781 to 0.896 and val accuracy 0.4 points lower. Train loss kept falling (0.58 to 0.33), so epoch 3 is overfitting. Caveat: the test accuracy above is measured on the final (epoch-3) model, not the best epoch.

**Confusion matrix (A100 run; rows = true, columns = predicted; tweet_eval labels 0/1/2 = negative/neutral/positive):**
| | pred 0 | pred 1 | pred 2 |
|---|---|---|---|
| true 0 | 80 | 43 | 8 |
| true 1 | 33 | 253 | 46 |
| true 2 | 8 | 87 | 192 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 negative | 0.661 | 0.611 | **0.635 (worst)** | 131 |
| 1 neutral | 0.661 | 0.762 | 0.708 | 332 |
| 2 positive | 0.780 | 0.669 | 0.720 | 287 |

Worst class: 0 (negative). With only 131 test examples its F1 gap to the other classes is within noise. The dominant error is drift toward the middle class (87 positives and 43 negatives predicted neutral).
Five misclassified class-0 tweets (all predicted neutral), with diagnoses:
1. "@user @user aaron, if you take my Netflix and chill girl we are fighting tomorrow" - a joking threat; the negativity is playful and implicit, no negative words.
2. "This @user fine thing sounds like the ejection rule in CFB..." - sarcastic commentary on a sports rule; the complaint is implied.
3. "@user listed as star attraction for Aust-China event today. Rudd's office say he never committed..." - news-style report; negative stance is carried by the event, not the wording.
4. "@user @user Doomsday may be sooner than you think w/ Obama's signed Iran Deal..." - political sarcasm; needs world knowledge to read as negative.
5. "when Trayvon touched my shoulder in 7th period and tried to hide like it wasn't him..." - a joke whose negative reading depends on context outside the tweet.
Common cause: sarcasm, jokes and news-like text with no explicit sentiment words, so the model defaults to neutral.

**Save / reload / custom inference (T4, model saved with `save_pretrained`, reloaded in a fresh process):**
| Input | Prediction |
|---|---|
| I love this phone, best purchase ever | positive 0.99 |
| worst service I have ever had, never again | negative 0.94 |
| The meeting is at 5pm tomorrow | neutral 0.97 |
| lol great, another Monday | positive 0.98 (**wrong**: sarcasm, same failure type as the worst-class errors) |

## 2. Part A - Batch scaling to the memory wall (1 epoch per batch size, no eval)
| Batch | A100 samples/s | A100 peak MB | A100 s/epoch | T4 samples/s | T4 peak MB | T4 s/epoch |
|---|---|---|---|---|---|---|
| 8 | 160.6 | 2,145 | 21.8 | 37.0 | 2,144 | 94.5 |
| 16 | 191.5 | 2,537 | 18.2 | 41.8 | 2,537 | 83.4 |
| 32 | 210.0 | 3,827 | 16.6 | 44.8 | 3,824 | 77.8 |
| 64 | 225.4 | 6,282 | 15.3 | 45.9 | 6,254 | 75.3 |
| 128 | 236.7 | 11,292 | 14.6 | 46.3 | 11,216 | 74.7 |
| 256 | 242.5 | 21,192 | 13.7 | OOM | OOM | OOM |
| 512 | OOM | OOM | OOM | - | - | - |

Figure: `fig_partA_throughput_memory_both_cards.png` (throughput and peak memory vs batch size, both cards).
- **Largest batch that runs:** A100 256, T4 128.
- **Where throughput stops improving:** not the same on both cards. T4: gains are under 1% after batch 64 (45.9 to 46.3), so the T4 saturates at 32-64. A100: still +5% from 64 to 128 and +2.4% from 128 to 256, so it saturates around 128. The bigger GPU needs a bigger batch to fill its compute.
- **Ratio of max batch sizes:** 256 / 128 = **2.0**. **VRAM ratio:** 40,960 / 15,360 = **2.67**.
- **Why they differ:** (1) Batch sizes are powers of two, so "max batch" is only known to within a factor of 2: the true ceilings sit somewhere in 128-255 (T4) and 256-511 (A100). Both cards use about 77 MB per extra sample, so estimated true ceilings are roughly 170 (T4) and 490 (A100), a ratio of about 2.8, close to the VRAM ratio (an estimate from the measured slope, not a measurement). (2) Fixed costs (weights, gradients, Adam state about 1.8 GB, plus CUDA context and allocator overhead) do not scale with batch, so memory left for activations is not proportional to total VRAM.

## 3. Part B - Memory techniques, measured one at a time
**Table HW4.5.1 - Tesla T4 (all four measurements per technique, max batch from the OOM scan; 3-epoch run with eval at that batch)**
| Technique | Max batch | Throughput (samp/s) | Peak memory (GB) | Eval accuracy (test; val) |
|---|---|---|---|---|
| Baseline (HW4 settings) | 128 | 46.3 | 10.95 | 0.713 (0.664), measured at the HW4 batch 16 |
| Mixed precision (AMP, fp16) | 256 | 190.5 | 13.91 | 0.612 (0.591) |
| Gradient checkpointing | 512 | 34.5 | 7.58 | 0.577 (0.555) |
| Gradient accumulation (16 x 16) | effective 256 | 45.7 | 2.90 | 0.617 (0.583) |
| 8-bit optimiser (AdamW8bit) | 128 | 46.3 | 10.34 | 0.683 (0.636) |
| Best combination: AMP + 8-bit AdamW | 256 | 188.9 | 13.29 | 0.612 (0.593) |
| Best combination at the HW4 batch 16 | 16 | 126.3 | 1.62 | 0.697 (0.673) |

**Reading the accuracy column:** accuracy at a large batch is lower because 3 epochs at batch 256 give only 39 optimizer steps (654 at batch 16) with the same learning rate, so the model is undertrained; the technique is not the cause. Evidence: every config with 39 steps lands at 0.58-0.62 (AMP 0.612, accumulation 0.617, combination 0.612), while the same-step comparison below shows no real technique effect, and AMP + 8-bit at batch 16 keeps 0.697 against 0.713 baseline (a 1.6 point gap, about one standard error on 750 examples; one seed).

**Table HW4.5.1 - A100 40GB (measured at batch 16 only; per-technique max batch not measured on this card)**
| Technique | Max batch | Throughput (samp/s) | Peak memory (GB) | Eval accuracy (best val) |
|---|---|---|---|---|
| Baseline (HW4 settings) | 256 | 242.5 (194.3 in the 3-epoch run at batch 16) | 20.7 (2.48 at batch 16) | 0.700 test (val 0.684) |
| Mixed precision | not measured | 341.2 (batch 16) | 2.26 (batch 16) | val 0.664 |
| Gradient checkpointing | not measured | 148.1 (batch 16) | 2.07 (batch 16) | val 0.684 |
| Gradient accumulation (16 x 4) | effective 64 | 206.8 | 2.89 | val 0.667 |
| 8-bit optimiser | not measured | 169.6 (batch 16) | 1.88 (batch 16) | val 0.672 |
| Best combination (AMP + ckpt + 8-bit, batch 16) | not measured | 168.7 | 1.09 | val 0.672 |

**Effect of each technique (T4, same batch size):**
| Technique | Batch 16: speed / memory vs baseline | Batch 128: speed / memory vs baseline |
|---|---|---|
| AMP | 123.5 vs 41.8 samp/s (**2.95x**), 2,299 vs 2,537 MB (-9%) | 180.9 vs 46.3 (**3.9x**), 7,853 vs 11,216 MB (-30%) |
| Gradient checkpointing | 31.0 (**-26%**), 2,117 MB (-17%) | 33.4 (**-28%**), 2,913 MB (**-74%**) |
| 8-bit AdamW | 42.4 (+1%), 1,921 MB (-24%) | 46.3 (0%), 10,585 MB (-6%) |

- **Gradient checkpointing throughput cost:** 26-28% slower on the T4 (it re-runs the forward pass); on the A100 at batch 16 it was 24% slower (148.1 vs 194.3).
- **Gradient accumulation, same optimizer steps (T4):**
| Config | Effective batch | Optimizer steps | Wall-clock | Peak MB | Test acc (val) |
|---|---|---|---|---|---|
| 16 x 4 accumulated | 64 | 162 | 233.1 s | 2,968 | 0.672 (0.643) |
| plain batch 64 | 64 | 162 | 224.4 s | 6,254 | 0.687 (0.660) |
| 16 x 16 accumulated | 256 | 39 | 229.0 s | 2,968 | 0.617 (0.583) |
| plain batch 256 | 256 | - | OOM | OOM | - |
  Accumulation reaches the same optimizer steps in 3.9% more time with 53% less memory; the 1.5 point accuracy gap is within noise for one seed and 750 test examples. It reaches effective batch 256, which does not fit as a plain batch on the T4.
- **AMP + 8-bit at batch 16 (T4):** 82.9 s for the 3-epoch run versus 240.9 s baseline (**2.9x faster**), 1,662 MB versus 2,537 MB (-34%).

**Ranking: memory saved per unit of throughput lost (T4; AMP, checkpointing and 8-bit at batch 128 against the batch-128 baseline; accumulation at effective batch 64 against plain batch 64; loss floored at 1% so techniques that lose nothing stay finite)**
| Rank | Technique | Memory saved | Throughput change | Saved % per lost % |
|---|---|---|---|---|
| 1 | Mixed precision | 30.0% | +291% (nothing lost) | 30.0 |
| 2 | Gradient accumulation | 52.5% | -2.8% | 18.7 |
| 3 | 8-bit optimiser | 5.6% | 0% | 5.6 (24% saved at batch 16) |
| 4 | Gradient checkpointing | 74.0% | -27.9% | 2.7 |

**Apply first:** AMP (less memory and much faster on tensor-core GPUs; 3-4x here). **Not bother on a 32 GB card:** gradient checkpointing. BERT-base at batch 256 peaked at 21.2 GB on the A100, so it fits in 32 GB without paying a 25-28% slowdown; 8-bit AdamW and accumulation are also unnecessary there unless the batch is pushed far beyond 256.

## 4. Part C
Part C skipped (optional).

## 5. Part D - Serving, revisited (Qwen2.5-0.5B-Instruct, fp16, 128 new tokens, 64 prompts, gpu_util 0.80)
**Table HW4.1 / Part D, Tesla T4 (HF side is the `transformers.pipeline` text-generation baseline at batch 1):**
| Configuration | Tokens/sec (per request) | p50 latency (ms) | p95 latency (ms) | Total tok/s |
|---|---|---|---|---|
| HF pipeline, batch 1 | 34.5 | 3,580 | 4,203 | 34.3 |
| vLLM, batch 1 | 158.7 | 793 | 881 | 158.3 |
| vLLM, batch 8 | 144.9 | 880 | 902 | 1,159.1 |
| vLLM, batch 32 | 105.0 | 1,220 | 1,220 | 3,358.5 |

**A100 40GB** (HF row is the earlier custom `generate` loop, not the pipeline:
| Configuration | Tokens/sec (per request) | p50 latency (ms) | p95 latency (ms) | Total tok/s |
|---|---|---|---|---|
| HF generate, batch 1 | 35.2 | 3,620 | 3,710 | 35.2 |
| vLLM, batch 1 | 401.7 | 320 | 340 | 401.3 |
| vLLM, batch 8 | 338.2 | 380 | 390 | 2,705.2 |
| vLLM, batch 32 | 274.9 | 470 | 470 | 8,796.6 |

Note: vLLM's offline API returns latency per `generate` call, so all requests in a batch share that latency (p50 is nearly equal to p95).

**Latency versus throughput (answer):** In my measurements there is no batch size, up to 32, where vLLM's per-request latency is worse than the HF batch-1 baseline: on the T4 it grows from 793 ms (batch 1) to 1,220 ms (batch 32), still far below the baseline's 3,580 ms, while total throughput rises from 158 to 3,358 tok/s (21x). Extrapolating the roughly 14 ms of extra latency per added request, latency would only reach the baseline near batch 200 (an extrapolation, not a measurement). Throughput keeps improving the whole way, but per-request speed falls (159 to 105 tok/s per request). For an interactive application, per-request latency matters, because each user waits on their own response; total throughput matters for offline or batch workloads and for cost per token.

**KV cache memory versus context length (batch 8, fp16, measured by prefilling and counting bytes in the returned cache):**
| Context | 512 | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 |
|---|---|---|---|---|---|---|
| KV measured (MiB, T4) | 48 | 96 | 192 | 384 | 768 | 1,536 |
| Formula (MiB) | 48 | 96 | 192 | 384 | 768 | 1,536 |
Formula: 2 x 24 layers x 2 KV heads x 64 head_dim x 2 bytes = 12,288 B/token. The measurement matches exactly and grows linearly with context.

**Longest context at batch 8 before exhausting memory:** vLLM's own log reports KV capacity of 868,016 tokens on the T4 (9.93 GiB) and 2,588,384 tokens on the A100 (29.62 GiB) at gpu_util 0.80. Split over 8 sequences that is about **108,500 tokens per sequence on the T4 and about 323,500 on the A100**. Both exceed the model's own 32,768-token limit, so on both cards the usable longest context at batch 8 is **32,768 tokens, set by the model, not by memory**. (My forced-memory probe at gpu_util 0.08 on the T4 failed at every length because the engine could not start in 1.2 GB; it says nothing about the KV limit, so I report the engine-log capacity instead.)

**Comparison with my earlier (HW4.a-era) T4 run of the same benchmark:**
| T4 total tok/s | Earlier | Now | Change |
|---|---|---|---|
| vLLM batch 1 | 161.2 | 158.3 | -1.8% |
| vLLM batch 8 | 1,203.4 | 1,159.1 | -3.7% |
| vLLM batch 32 | 3,450.0 | 3,358.5 | -2.7% |
| HF batch 1 | 26.1 | 34.5 | +32% |
vLLM differences are within run-to-run noise (same card, same engine). The HF change is a **measurement change, not hardware**: the baseline is now the standard `transformers.pipeline` instead of my earlier hand-written generate loop with different token counting. The A100 versus T4 gap (402 vs 159 tok/s at vLLM batch 1, 2.5x) is hardware (memory bandwidth). HF batch 1 is almost identical on both cards (35 tok/s): it is limited by Python and kernel-launch overhead, not GPU speed.

**vLLM with another model and my own prompt (Qwen2.5-1.5B-Instruct, T4, gpu_util 0.30):** prompt "Explain in two sentences why a bigger batch size can make GPU training faster but use more memory." returned a correct two-sentence answer (more parallel data per pass, but more activations held in memory). Prompt "Write a one-line tweet that sounds negative but is actually a joke." returned a tweet about a "great deal" that did not follow the instruction (not negative-sounding), a small-model limitation.

## 6. Part E - Analysis and summary
**GPU-hours for one full fine-tune (3 epochs, 3,488 training samples per epoch = 3,500 rounded to full batches of 16):**
GPU-hours = (samples per epoch x epochs) / (samples per second) / 3600 = 10,464 / 43.4 / 3600 = **0.067 GPU-h (241 s) on the T4**; on the A100, 10,464 / 190 / 3600 = **0.015 GPU-h (55 s)**. A 20-run sweep would be about 1.3 GPU-h on the T4 and 0.3 GPU-h on the A100.

**Single configuration I would use for a full fine-tune:** AMP (fp16) + 8-bit AdamW at the HW4 batch size 16, otherwise HW4 settings. On the T4 it ran in 82.9 s instead of 240.9 s (2.9x), peak memory 1,662 MB instead of 2,537 MB (-34%), test accuracy 0.697 versus 0.713 (within noise on one seed). A large batch is not worth it: AMP at batch 256 is about 50% faster than at batch 16 but undertrains (0.612) at the same learning rate. On a 24-32 GB card memory is not binding, so AMP alone captures most of the gain.

**HW4.5 summary (do not use as an HW5 baseline):**
| | Best configuration | Value |
|---|---|---|
| Lowest peak memory at HW4 batch (T4) | AMP + 8-bit AdamW, batch 16 | 1,662 MB (A100: AMP + ckpt + 8-bit 1,112 MB) |
| Highest throughput | T4: AMP, batch 256 | 190.5 samples/s (A100: baseline batch 256 242.5; AMP batch 16 341.2) |
| Shortest wall-clock (3 epochs, HW4 batch 16) | T4: AMP + 8-bit | 82.9 s (baseline 240.9 s); A100: AMP about 31 s (10,464 / 341.2; baseline 55.0 s) |

Cost in Colab units (third-party observed rates, check your resources panel): T4 about 1.19 units/h and A100 about 5.4 units/h, so both come to roughly 0.08 units per fine-tune (the A100 is about 4.3x faster).


