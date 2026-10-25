# Metrics: Mini DDPM -> DDIM on MNIST

**Config:** T = 100, linear beta 1e-4 -> 0.02 (assigned), batch 128, 50 epochs, Adam (lr 2e-4), EMA decay 0.999, GPU (CUDA).
**Model:** U-Net-style CNN, 8 conv layers, SiLU, GroupNorm, sinusoidal timestep embedding, 1.82M parameters.
**Schedule check:** alpha_bar_1 = 0.9999, alpha_bar_T = 0.3636 (not ~0, so x_T is not N(0, I)).

## Training
| Metric | Value |
|---|---|
| Epoch-1 loss (noise MSE) | 0.1650 |
| Epoch-10-ish loss (epoch 11) | 0.0627 |
| Final loss (epoch 50) | 0.0548 |
| Total training time | 1311 s (~22 min) |
| Iterations per epoch | 468 |

## Sampling (16 images, same starting noise x_T)
| Sampler | Network evals | Time (s) | Speed-up vs DDPM | MSE vs DDIM-100 reference |
|---|---|---|---|---|
| DDPM | 100 | 0.31 | 1.0x | n/a (stochastic) |
| DDIM-50 | 50 | 0.19 | 1.6x | 0.0247 |
| DDIM-20 | 20 | 0.06 | 5.2x | 0.2159 |

## Determinism / stochasticity
| Check | MSE |
|---|---|
| DDIM-50 vs DDIM-50 rerun, same noise | 0.000000 (deterministic) |
| DDPM vs DDPM, same x_T, different seeds | 0.3714 (stochastic) |

## Notes
- MSE vs the DDIM-100 reference measures agreement between samplers, not image quality. DDIM-50 agrees closely with DDIM-100 partly because both are noisy in the same way.
- Timings are single runs on a small batch (16), so they are rough; fixed overhead makes the speed-up smaller than the step ratio.
- No quantitative sample-quality metric (FID etc.) was computed; quality is assessed visually.
