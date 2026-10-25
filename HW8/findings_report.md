# Findings Report: Mini Diffusion Model (DDPM -> DDIM) on MNIST

## Setup
T = 100 diffusion steps, linear beta 1e-4 -> 0.02, batch size 128, 50 epochs. Noise predictor: U-Net-style CNN with 8 conv layers (SiLU, GroupNorm, skip connections), sinusoidal timestep embedding added as per-channel bias, output same shape as input; trained with the simple loss E||eps - eps_theta(x_t, t)||^2. EMA weights (decay 0.999) used for sampling. DDPM uses ancestral sampling with sigma_t^2 = beta_t over 100 steps; DDIM (eta = 0) reuses the same trained network with a strided subset of 50 or 20 timesteps.

## Results
- **Training:** loss fell from 0.165 (epoch 1) to 0.0548 (epoch 50), still decreasing slowly. Training took about 22 minutes.
- **Forward diffusion:** images degrade smoothly, but at t = T the image is not fully destroyed: alpha_bar_T = 0.364.
- **Stochasticity:** DDIM is deterministic (rerun MSE = 0). DDPM from the same x_T gives different images (MSE 0.371).
- **Speed:** DDIM-50 is 1.6x and DDIM-20 about 5x faster than 100-step DDPM (0.19 s and 0.06 s vs 0.31 s for 16 images).
- **Quality:** with the assigned schedule none of the samplers produces clean digits. DDPM yields smooth, thick, swirly shapes; DDIM-50 is very noisy; DDIM-20 is noisy but more structured. Fewer DDIM steps looked better here, not worse.
- **MSE vs DDIM-100:** 0.025 (DDIM-50) and 0.216 (DDIM-20). This measures agreement between samplers, not quality: DDIM-50 is close to DDIM-100 largely because both are noisy in the same way.

## Interpretation
The likely cause of the poor samples is a train/sample mismatch: with only 100 steps and beta_end = 0.02, x_T still keeps about 36% of the signal in alpha_bar, but sampling starts from pure N(0, I), an input the network never saw at t = T. Errors from the first steps carry through the chain. DDPM's per-step noise may let it partly recover, while deterministic DDIM carries errors forward, consistent with DDIM looking noisier. This was not isolated by a controlled experiment in this submission; a larger T or larger final beta (making alpha_bar_T ~ 0) is the natural fix, but the assignment fixes beta_end = 0.02.

## Takeaways
1. DDIM gives faster, deterministic sampling from the same trained model (x_T acts as a latent code).
2. DDPM is slower but more forgiving of predictor error and gives different samples from the same noise.
3. Both samplers depend on the noise schedule reaching near-pure noise at t = T; the assigned range does not do so at T = 100, which dominates sample quality here.

## Limitations
Single run and seed; timings measured on 16 images; quality judged visually without FID or similar metrics; model is small (1.82M parameters).
