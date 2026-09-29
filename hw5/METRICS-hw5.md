# METRICS.md - LoRA fine-tuning of Flan-T5-small on DialogSum

Student ID (last 4): 6638 | SEED=6638 | SLICE=638 | HP_ID=2
Hardware: Google Colab, NVIDIA Tesla T4 | Model: `google/flan-t5-small` | Dataset: `neil-code/dialogsum-test`
Training subset: 638 dialogues (seeded shuffle of train split) | Eval: first 100 test dialogues (499 in test split)

## Table 1 - Baseline vs. LoRA (rank experiment)

| Model | LoRA r | alpha | Dropout | Total params | Trainable params | Trainable % | Final train loss | Mean train loss | Train time | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (no fine-tuning) | - | - | - | 76,961,152* | 0 | 0% | - | - | - | 0.1483 | 0.0474 | 0.1364 |
| LoRA r=4 | 4 | 8 | 0.05 | 77,133,184 | 172,032 | 0.223% | 1.5341 | 1.5895 | 2.0 min | 0.3443 | 0.1058 | 0.2815 |
| LoRA r=16 | 16 | 32 | 0.05 | 77,649,280 | 688,128 | 0.886% | 1.4504 | 1.5184 | 2.0 min | 0.3498 | 0.1056 | 0.2831 |

*Baseline total = r=4 total minus r=4 trainable (77,133,184 - 172,032), i.e. the frozen base model.

LoRA config: target modules `q`, `v` (attention), bias none, task SEQ_2_SEQ_LM. Training: 5 epochs, batch size 8, lr 1e-3, beam search (4 beams) for generation.

## Observations
- Both LoRA runs roughly 2.4x ROUGE-1 and 2x ROUGE-L over the baseline while training under 1% of parameters.
- r=16 has exactly 4x the trainable parameters of r=4 and slightly lower training loss; ROUGE is nearly identical (within noise for 100 eval dialogues, one seed).
- Qualitative: r=16 produced a coherent second sentence on Dialogue 2 where r=4 repeated itself.

Traceability: every value above appears in `RUN_LOG.txt` and in the executed notebook `DATA266_LoRA_DialogSum_final.ipynb`.
