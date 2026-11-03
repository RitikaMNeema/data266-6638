# GPU-hour record - HW4.5
One full fine-tune = 3,488 samples/epoch x 3 epochs = 10,464 samples. GPU-hours = 10,464 / (samples/s) / 3600.
| Card | Samples/s | One fine-tune | GPU-hours |
|---|---|---|---|
| T4 | 43.4 | 241 s | 0.067 |
| A100 | 190 | 55 s | 0.015 |
Colab units (third-party observed rates; check the Colab resources panel): T4 ~1.19 units/h, A100 ~5.4 units/h, about 0.08 units per fine-tune on either card.
total GPU-hours - 3 hours