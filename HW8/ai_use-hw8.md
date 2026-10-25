# AI Use Disclosure

**Tool:** Claude (Anthropic).

**What Claude did**

- Suggested later changes after seeing run outputs: a deeper U-Net with skip connections and EMA weights (to improve sample quality), and a diagnosis of why the first run produced poor samples (alpha_bar_T = 0.36 with beta_end = 0.02 and T = 100).
- Drafted the discussion text and the markdown explanations in the notebook, using the numbers from my executed runs.
- Helped prepare these supporting files (metrics, run logs, findings report).

**What I did**

- Wrote the initial notebook code: forward diffusion, timestep-conditioned CNN noise predictor, training loop, DDPM and DDIM samplers, plots and comparison cells.
- Ran all code myself on Colab (GPU) and supplied the executed notebooks and screenshots; every number and image in the submission comes from those runs. Claude did not execute the notebook itself.
- Chose the final configuration: after consulting the TA, I use the required beta_end = 0.02.
- Reviewed the results and the written discussion against my generated images before submitting.

**Verification and limits**
- Claude had no GPU or PyTorch in its environment, so the code was checked for syntax and shapes but only validated by my runs.
- The explanation for poor sample quality (train/sample mismatch from alpha_bar_T = 0.36) is a hypothesis; a run with beta_end = 0.2 earlier gave clearly better digits, which supports it, but that run is not part of this submission.
