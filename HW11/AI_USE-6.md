# AI Use Disclosure — DATA 266 CLIP Assignment

**Student:** Ritika Mukesh Neema
**Assignment:** CLIP caption–image similarity (Part 1) and zero-shot CIFAR-10 classification (Part 2)
**AI tool used:** Claude (Anthropic), via claude.ai

## What the AI was used for

| Area | How AI was used |
|---|---|
| Notebook code | Generated the initial Colab notebook: CLIP loading, image/text embedding helpers, 100-per-class sampling, cosine similarity, prompt-ensemble zero-shot classifier, evaluation (accuracy, per-class accuracy, confusion matrix, classification report), plots, and code that writes `METRICS.md`, `metrics.json` and `RUN_LOG.txt`. |
| Testing | The AI did a dry run of the notebook with a tiny randomly initialized CLIP model and synthetic images to catch code errors before I ran it. That run produced no real results. |
| Findings text | After I ran the notebook, the AI rewrote the interpretation cells to match my actual results (e.g. 89.31% top-1, deer→horse as the largest confusion). |
| Report | Drafted the PDF report from my notebook outputs and figures. |

## What I did

- Ran the full notebook myself in Google Colab (A100 GPU, seed 6638). Every number and figure in the notebook, report and `outputs/` folder comes from that run.
- Reviewed the code, outputs and figures, and checked that the findings and report match the printed results.
- Decided what to submit and how to organize the deliverables.

## Verification

- Every metric in the report was checked against the notebook output and `METRICS.md` (Part 1: Precision@100 0.76, ROC-AUC 0.9626; Part 2: top-1 88.31% single template, 89.31% ensemble).
- The pretrained model is `openai/clip-vit-base-patch32`, used as is, with no fine-tuning. The dataset is the torchvision CIFAR-10 test split.

I understand the methods used and can explain the code and results.
