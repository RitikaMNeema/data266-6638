# METRICS — STL-10 SSL (SID4=6638, SEED=6638)

Backbone: ResNet-18, no pretrained weights. Labeled subset: 500 images (10%, stratified). Test set: 8000 images.

| Part | Method | Pretrain epochs | Linear/Train epochs | Test accuracy |
|---|---|---|---|---|
| A | Supervised (10% labels) | – | 15 | 38.12% |
| B | Rotation SSL + linear probe | 15 | 20 | 49.31% |
| C | SimCLR (τ=0.2) + linear probe | 20 | 20 | 53.80% |

Rotation pretext accuracy (last epoch): 83.2%  |  SimCLR unlabeled images: 20000

## Part D — kNN precision@5 (cosine, whole test set)
- Supervised: 36.36%
- Rotation SSL: 40.96%
- SimCLR: 47.46%

Query test indices: [3808, 2548, 3442]
