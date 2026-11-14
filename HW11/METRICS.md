# METRICS — CLIP zero-shot (openai/clip-vit-base-patch32)

## Part 1 — caption similarity
- Caption: "a photo of a dog" vs 1000 images (100/class)
- Precision@100: 0.760
- ROC-AUC (dog vs rest): 0.9626

| Class | Mean cosine |
|---|---|
| dog | 0.2611 |
| cat | 0.2401 |
| deer | 0.2347 |
| frog | 0.2292 |
| bird | 0.2269 |
| horse | 0.2191 |
| airplane | 0.2027 |
| automobile | 0.1978 |
| truck | 0.1949 |
| ship | 0.1933 |

## Part 2 — zero-shot CIFAR-10 (10,000 test images)
- Top-1: single template 0.8831 | ensemble 0.8931
- Top-3: single template 0.9746 | ensemble 0.9767

| Class | Single template | Ensemble |
|---|---|---|
| airplane | 0.889 | 0.907 |
| automobile | 0.933 | 0.944 |
| bird | 0.897 | 0.882 |
| cat | 0.857 | 0.865 |
| deer | 0.770 | 0.801 |
| dog | 0.891 | 0.884 |
| frog | 0.715 | 0.767 |
| horse | 0.971 | 0.976 |
| ship | 0.959 | 0.964 |
| truck | 0.949 | 0.941 |
