# AI Use Disclosure

**Assignment:** DATA 266 HW11: GPTQ quantization (Part 1) and knowledge distillation (Part 2)
**Student:** Ritika Mukesh Neema
**AI tool used:** Claude (Anthropic), through the Claude app

## What the AI tool did

| Area | How AI was used |
|---|---|
| Notebook code | Wrote both Colab notebooks. **Part 1:** pinning the model revision, Wikitext-2 calibration and perplexity evaluation, the GPTQ quantize/save/reload flow, and the p50/p95 latency, peak-memory and on-disk measurements. **Part 2:** teacher download and extraction, a teacher label-order check, teacher-logit extraction for all 120k training samples, the distillation loss, the training loop, both sweeps, the shared teacher/student benchmark, per-class analysis, and example selection. |
| Testing | Before I ran anything, the AI ran both notebooks end to end on CPU with tiny randomly initialised models and synthetic data. For Part 1 this used a 2-layer OPT and real GPTQ quantization with `gptqmodel` on CPU. For Part 2 it used a tiny BERT teacher trained with deliberately scrambled label columns, to test the label-order check. These runs produced no real results. |
| Debugging | Diagnosed the errors I hit in Colab and changed the notebooks to fix them (see below). |
| Teacher file access | When the shared Drive link hit its download limit, the AI used my Google Drive connection to copy `teacher_model.7z` into my SJSU Drive. It then shared that copy view-only with the Gmail account my Colab session uses. |
| Write-ups | Wrote the findings cells in both notebooks, `METRICS.md`, `RUN_LOG.txt` (copied line by line from the notebook logs), this file, and the PDF report, all from the outputs of my executed notebooks. |

## What I did

- Provided the assignment specification and the Step 0 parameters from Assignment 1 (SEED = 6638).
- Ran Part 1 on a Colab L4 GPU and Part 2 on a Colab T4 GPU, end to end, and returned the executed notebooks.
- Reported the errors from my Colab runs and applied the fixes in my sessions.
- Reviewed the outputs, findings and deliverables before submission.

## Where the AI's output was wrong or incomplete

1. **Assumed values.** In the assignment text I pasted, the temperature list, the α values and the base model name came through blank. The AI chose T ∈ {1, 2, 4, 8}, α = 0.5 for the temperature sweep, α ∈ {0.25, 0.5, 0.75} for the α sweep, and `facebook/opt-125m`, and flagged them as assumptions to check against Canvas.
2. **First dependency fix failed.** Installing `gptqmodel` upgraded PyTorch, after which Colab's preinstalled `torchaudio` (built for a different CUDA version) broke `import transformers`. The AI's first fix was to uninstall `torchaudio`/`torchvision`. That led to a new `ModuleNotFoundError`, because Transformers had already decided `torchaudio` was installed when it was first imported. The working fix was to reinstall `torchaudio`/`torchvision` builds matching the new PyTorch and then restart the runtime.
3. **Wrong Drive path.** The first fallback for the teacher download assumed the copy would appear at `/content/drive/MyDrive/`. My Colab session is signed in to a different Google account than the one that owns the copy, so the file was not there. The working fix downloads by file ID through the authenticated Drive API, after sharing the copy with that account.
4. **Latency measurement flaw.** The benchmark warmed up with batched inputs but not single examples before timing single-example (batch-1) latency. This probably inflated the teacher's batch-1 timing (p50 29.8 ms, p95 56.9 ms). The analysis therefore uses the batched 2.0× throughput as the reliable speedup and says so.
5. **Caught before my run:** in the AI's own smoke test, the teacher benchmark logged accuracy before reordering the label columns. The AI fixed it before handing over the notebook.

## Verification

- Every number in the report, `METRICS.md` and the findings cells comes from the outputs of my executed notebooks; none were estimated or typed in by hand. `RUN_LOG.txt` is copied line by line from the notebooks' `log()` output.
- Sanity checks that passed in my runs:
  - The FP16 OPT-125M perplexity (27.66) matches the value reported in the GPTQ paper.
  - The teacher scored 98.7% on the 2,000-sample label-order check.
  - The repeated T=4, α=0.5 student reproduced its first run exactly.
  - The quantized model's perplexity after reloading from disk is in the expected range.
- Best T and α were chosen on a held-out validation split, not on the test set.

## Limitations of AI assistance

- The AI could not run either notebook on a GPU or download the real models and data. All results come from my Colab runs.
- Sweep differences are within about one standard error, so the "best" T and α are not meaningfully better than the alternatives. The analysis states this.

I understand the methods used and can explain the code and results.
