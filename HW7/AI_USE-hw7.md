# AI Use Disclosure — GANs (Theory) and Autoencoders on FashionMNIST

**Student:** Ritika Mukesh Neema | **Course:** DATA 266 | **Tool:** Claude (Anthropic), Claude Sonnet 5.5

## What the AI did
- **Notebook code (Part 2):**  Step 0 parameter cell, data loading, 
- **Written answers (Part 1):** Claude drafted the GAN architecture, mode-collapse and variant answers (DCGAN, Conditional GAN, CycleGAN).
- **Analysis (Part 2.9):** Claude drafted the analysis text from the run outputs.
- **Reports:** Claude generated `METRICS.md`, `RUN_LOG.txt` (reconstructed from the notebook's printed output), this file and the PDF report.

## What I did
- the FC-AE and VAE models, training and evaluation loops, and the plotting, t-SNE, linear-probe and interpolation code.
- Ran the notebook end to end on Google Colab (Tesla T4 GPU). All reported numbers and figures come from that run..

## Verification of AI output
- Claude smoke-tested the code on a CPU with random fake data, then on real FashionMNIST on a CPU, before I ran it on Colab. Both Claude's CPU run and my Colab run gave matching conclusions (FC-AE test MSE 0.00841 vs 0.00843; VAE 0.01518 vs 0.01510).
- Claude viewed the generated figures and checked each written claim in the analysis against them.

## Example of AI output that was wrong and how it was fixed
1. **Analysis numbers did not match my run.** Claude's first analysis text quoted numbers from its own CPU run (e.g. FC-AE MSE 0.00841, VAE KL about 12.01, probe accuracy 80.8%). My Colab run gave slightly different values (0.00843, 11.93, 80.7%). Claude noticed the mismatch when I uploaded my executed notebook and rewrote the analysis with my exact values.
2. **Unsupported claim removed.** A draft sentence said the VAE's reconstruction/KL split "shows most of the 32 dimensions carry modest information." Nothing in the notebook measures per-dimension information, so it was removed before delivery.
3. **Claims toned down to match my plots.** Claude had described the VAE's t-SNE clusters as clearly more continuous and the VAE interpolation as "plausible blends." On my Colab plots the two t-SNE maps look broadly similar and the VAE's midpoint is a blurry blob, so the text was softened.
4. **Environment issue (not code):** installing PyTorch from the official CPU index failed in Claude's sandbox because the host was blocked; Claude installed it from PyPI instead. This did not affect my Colab run.

## Statement
I understand and can explain the submitted code and answers. The AI's contribution is described above.
