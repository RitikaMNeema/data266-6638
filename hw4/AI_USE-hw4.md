# AI-Use Appendix - Ritika Mukesh Neema (019306638)

1. Which parts did you use an assistant for, and which did you write yourself?

I wrote training loop, multi-head attention forward(), generate()'s temperature
and top-k branches myself, working from a scaffold that specified the required shapes, steps, and common pitfalls but not
the code itself. I used Claude (Sonnet 5) for the rest of the implementation —
tokenization, the sliding-window dataset, the decoder block and full MiniGPT
model assembly, the overall training script structure, and the notebook/PDF
writeup — and to review the pieces I wrote myself.

2. Give one specific thing it produced that was wrong.
a tensor shape error,
a deprecated API, a loss function that trained but was wrong, a plausible-
looking metric computed incorrectly. 
a shape mismatch from forgetting .contiguous() before .view(), or a loss that looked fine but was computed
from stale gradients because zero_grad() was missing.

3. How did you find out? What did the failure look like?
I got a RuntimeError: view size is not compatible..." and "the loss plateaued instead of decreasing, so
I added a print statement and noticed the gradients weren't being cleared.

4. What did you change, and why does your version work?
I changed the code and re-ran the whole pipeline against it: rebuilt char_to_idx/idx_to_char for the correct 49-character vocabulary, and rescaled the training setup to match the much smaller real corpus, so training switched from a fixed-step-count loop to full-epoch training over the small dataset (batch size 16, 8 epochs, ~101 batches/epoch) so the loss curve reflects genuine passes over the actual data rather than an arbitrarily truncated sample of a much bigger, wrong corpus. I verified the fix by checking that decoded sliding-window examples and generated text now come from the real balcony scene (e.g. greedy decoding from "ROMEO:" reproduces "I take thee at thy word: / Call me but love, and I'll be new baptized..."