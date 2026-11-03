# AI_USE.md

**Assignment:** Latent Diffusion and Stable Diffusion (DATA 266)
**Student:** Ritika Mukesh Neema
**Seed:** 6638

## Tool used
Claude (Anthropic), Sonnet 5.5, via Claude Code / claude.ai.

## What the AI did
- Wrote the first version of the Colab notebook `latent_diffusion_fashionmnist.ipynb`: Step 0 setup and seeding, VAE encoding and reconstruction check, latent forward diffusion, the U-Net (base 64, multipliers (1, 2, 4), self-attention at 16x16 and bottleneck, time + label embeddings), classifier-free-guidance training and sampling.
- Smoke-tested the U-Net on CPU (output shape, one backward pass, guidance batch) and syntax-checked every notebook cell.
- Estimated the A100 runtime and suggested TF32 settings.
- Drafted the written analysis (`Latent_Diffusion_FashionMNIST_Analysis.docx`) from the outputs of my own runs and the figures they produced.
- Drafted this disclosure.

## What I did
- Ran the notebook end to end in Google Colab on a T4 and again on an A100-SXM4-40GB (GPU, torch 2.11).
- Provided the assignment requirements, the Step 0 convention (SEED=6638) and the run outputs.
- Reviewed the results and figures: reconstruction MSE 0.00077, final training loss 0.0932, 64-image sample grid, guidance comparison at w = 0, 1, 3, 7.5.

## Choices not specified by the assignment
- Guidance scale for the main grid: w = 3.0 (the prompt left the value blank).
- Input images resized from 28x28 to 128x128 so the VAE latent is 4x16x16, matching the 16x16 attention requirement.
- Label embedding is added to the time embedding (per the spec); no cross-attention layer was added.
- AdamW with gradient clipping at 1.0 and batch size 128; DDPM ancestral sampling with 1000 steps; no EMA.

## Verification
- Reported numbers (MSE, loss, training time, latent std) come directly from the Colab run logs and `METRICS.md`.
- Claims about sample quality are from visual inspection only; no FID or classifier-based metric was computed.
- Explanations of why bags and sandals are weak are hypotheses, not tested.

## Limitations of the AI's work
- The AI did not run the full training or sampling itself; all reported results are from my Colab runs.
- Generated text and code should be, and were, checked against the actual outputs before submission.
