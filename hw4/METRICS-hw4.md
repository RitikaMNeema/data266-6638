Please fill in the table yourself. You can consult the [Markdown syntax guide](https://www.markdownguide.org/extended-syntax/#tables) on how to create a table in Markdown.

## Dataset

| Metric | Value |
|---|---|
| Source file | Shakespeare.txt (Romeo & Juliet balcony scene) |
| Total characters | 1,738 |
| Vocabulary size | 49 |
| Sequence length (block size) | 128 |
| Sliding-window training pairs | 1,610 |

## Model architecture

| Metric | Value |
|---|---|
| Hidden dimension (d_model) | 128 |
| Attention heads | 4 |
| Decoder blocks | 4 |
| Feed-forward hidden size | 512 (4 × d_model) |
| Total parameters | 822,272 |

## Training configuration

| Metric | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 3e-4 |
| Batch size | 16 |
| Epochs | 8 |
| Batches per epoch | 101 |
| Total training time (CPU) | ~122s |

## Training loss by epoch

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Loss | 2.691 | 2.188 | 1.908 | 1.556 | 1.109 | 0.733 | 0.496 | 0.367 |

## Generation settings

| Metric | Value |
|---|---|
| Prompt | "ROMEO:" |
| New tokens generated per sample | 300 |
| Temperature values compared | 0.5, 1.0, 1.5 |
| Top-k values compared | 5, 20 |
| Random seed | 6638 (student ID 019306638) |
