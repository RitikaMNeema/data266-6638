# METRICS — FC-AE vs VAE on FashionMNIST

SID4=6638 | SEED=6638 | SLICE=638 | HP_ID=2 | CLS_A=8 | CLS_B=5
Config: latent=32, hidden=512-256, Adam lr=0.001, batch=128, epochs=20

| Metric | FC-AE | VAE |
|---|---|---|
| Parameters | 1,083,696 | 1,091,920 |
| Final train loss | 0.00818 (MSE/px) | 238.19 (-ELBO/img) |
| Final test loss | 0.00843 (MSE/px) | 240.33 (-ELBO/img) |
| Test recon MSE/pixel | 0.00843 | 0.01510 |
| Test PSNR (dB) | 20.74 | 18.21 |
| VAE test BCE / KL per image | - | 228.39 / 11.93 |
| Linear-probe test accuracy (32-d latent) | 0.8267 | 0.8068 |
| Training time (s) | 190.5 | 196.5 |
