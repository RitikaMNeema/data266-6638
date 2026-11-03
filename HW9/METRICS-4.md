# METRICS
- Seed: 6638
- VAE: stabilityai/sd-vae-ft-mse, scale=0.18215, latent 4x16x16 (input resized to 128x128)
- VAE reconstruction MSE (512 test imgs): 0.00077 on [-1,1] scale, 0.00019 on [0,1] scale
- Diffusion: T=1000, linear beta 0.0001..0.02
- U-Net params: 14.49M (base 64, mults (1,2,4), attention at 16x16 + bottleneck)
- Training: 50 epochs, LR 2e-05, batch 128, label drop 0.1; final loss 0.0932; time 17.9 min
- Sampling: DDPM 1000 steps, guidance scale w=3.0
