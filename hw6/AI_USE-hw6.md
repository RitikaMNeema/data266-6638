# AI_USE.md — STL-10 Self-Supervised & Contrastive Learning (SID4 6638)

## Tool used
Claude (Anthropic), through a chat/agent session. No other AI tool was used for this assignment.

## What the AI did
- Wrote all notebook code: GPU-side augmentation (random resized crop, flip, colour jitter, grayscale), the ResNet-18 setup, Part A supervised training, Part B rotation pretraining and frozen linear probe, Part C SimCLR (NT-Xent, tau = 0.2, projection head) and frozen linear probe, Part D cosine nearest-neighbour retrieval and the kNN precision@5 metric.

## What I did
- Chose the hyperparameters (epochs, batch sizes, learning rates, 20k-image SimCLR subset, query-image selection by seed).
- Ran the full notebook on a Colab T4 GPU and obtained all reported numbers (A 38.12%, B 49.31%, C 53.80%; kNN precision@5 36.36 / 40.96 / 47.46).
- Supplied the assignment spec and the Step 0 parameters (SID4 = 6638) from Assignment 1.
- Wrote the Analysis section of the notebook and the PDF report from the results I ran.

## Checks that were done
- The AI ran a small dry-run of the code on random CPU data before I used it on Colab. It ran without errors.
- The code asserts that the encoder weights are unchanged after each linear probe (frozen-encoder requirement); this passed in both Part B and Part C.
- The run on real STL-10 completed without errors, with the 500-image subset balanced at 50 per class.
- The same-class neighbour counts quoted in the analysis (2/15, 3/15, 8/15) were counted by the AI by eye from the figure, not computed in code; the whole-test-set kNN precision@5 is the computed number.

## Example where AI output was flawed
No code crashed or produced wrong output in the real run, so I do not have a failing-code example like Assignment 1's. The flaws were of these kinds:

1. **Weak baseline settings.** The AI picked batch size 64 for Part A. With 500 images that is about 8 steps per epoch, so 15 epochs is only ~120 gradient updates, and the loss was still falling at the end (2.33 to 0.88). The 38.12% baseline is therefore undertrained, which makes the gap to the SSL models look larger than a well-tuned supervised baseline would give. This is stated in the report.
2. **Assumed spec details.** The AI could not open the Canvas assignment or the demo notebook, so it assumed the "4 augmentations utilized in the demo" are random resized crop, horizontal flip, colour jitter and random grayscale, and chose the optimiser settings itself.
3. **A wrong statement in chat.** The AI first said issues in Parts B/C had been caught by its dry-run. That was incorrect: the dry-run found no bugs. It corrected this when asked what to put in this file.

## Limitations to be aware of
Single run, single seed; with only 500 labels, gaps between models could shift by a few points across seeds.
