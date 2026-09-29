# AI-Use Appendix - Ritika Mukesh Neema (SID 019306638)

1. Which parts did you use an assistant for, and which did you write yourself?

I used Claude (Anthropic) to draft the Colab notebook: the data preparation, baseline inference, LoRA setup with PEFT, the fine-tuning loop, the rank experiment (r=4 vs r=16), and the ROUGE evaluation. It also drafted the wording of the Step 6 comparison and the Step 7 findings, based on the outputs from my own run, and generated the METRICS.md and RUN_LOG.txt files from those outputs. I chose the SID-derived parameters (SEED, SLICE, etc.) from the course's Step 0, ran everything on Colab myself on a T4 GPU, fixed the runtime issue below, and checked the numbers and sample summaries against my run output.

2. Give one specific thing it produced that was wrong.

The notebook's install cell (`pip install transformers peft datasets accelerate evaluate rouge_score`) did not account for Colab's preinstalled `torchao`. Running the LoRA cell failed with:

ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported

3. How did you find out? What did the failure look like?

The cell that calls `get_peft_model` crashed with the ImportError above, raised from `peft/import_utils.py` (`is_torchao_available`). Steps 0-2 (data prep and baseline inference) had already run fine, so the failure only appeared when PEFT was first used.

4. What did you change, and why does your version work?

I added `!pip install -q -U torchao peft`, restarted the Colab session, and reran the notebook. Upgrading torchao to a version that meets PEFT's minimum requirement removes the version mismatch, and the restart is needed because the old torchao was already loaded in memory. After that, both LoRA runs (r=16 and r=4) completed.
