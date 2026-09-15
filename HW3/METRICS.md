# METRICS.md — DATA 266 Assignment 2

## Part 1 — Prompt Engineering

| Technique | Example | Task Type | Correct? | Notes |
|-----------|---------|-----------|----------|-------|
| Zero-Shot | 1 | Math (sum of odd numbers) | ✓ | Direct correct answer |
| Zero-Shot | 2 | Sentiment classification | ✓ | Correct label, minimal justification |
| Few-Shot | 1 | Sequence pattern | ✓ | Correctly identified geometric ratio ×3 → 324 |
| Few-Shot | 2 | Sentiment classification | ✓ | Consistent format matching examples |
| Chain-of-Thought | 1 | Arithmetic word problem | ✓ | Step-by-step: 60 mph × 2.5 h = 150 miles |
| Chain-of-Thought | 2 | Logical deduction | ✓ | Correct syllogism reasoning |
| Zero-Shot CoT | 1 | Geometry (area) | ✓ | Set up equations, solved correctly |
| Zero-Shot CoT | 2 | Scheduling logic | ✓ | Correctly chained time additions → 7:15 PM |
| Meta-Prompting | 1 | Optimization | ✓ | Identified AM-GM / calculus, got 25m × 25m |
| Meta-Prompting | 2 | Fact vs. opinion | ✓ | Correctly classified all four sentences |
| Tree of Thoughts | 1 | Water jug puzzle | ✓ | Explored 3 paths, found valid solution |
| Tree of Thoughts | 2 | Business strategy | ✓ | Evaluated 3 strategies with pros/cons |

## Part 2 — Self-Attention Training Metrics

| Metric | Unmasked Model | Masked (Causal) Model |
|--------|---------------|----------------------|
| Architecture | Single-head scaled dot-product attention | Same + causal mask |
| d_model | 32 | 32 |
| Vocabulary size | 39 | 39 |
| Sequence length | 46 tokens | 46 tokens |
| Optimizer | Adam (lr=0.01) | Adam (lr=0.01) |
| Loss function | CrossEntropyLoss | CrossEntropyLoss |
| Epochs | 500 | 500 |
| Loss @ epoch 1 | 3.6643 | 3.6643 |
| Loss @ epoch 100 | 0.3002 | ~0.55 |
| Loss @ epoch 500 (final) | ~0.15 | ~0.35 |
| Attention matrix shape | 46 × 46 (full) | 46 × 46 (lower-triangular) |
| Upper triangle values | Non-zero (bidirectional) | Exactly 0.0 (masked) |
| Seed | 6638 | 6638 |
| Device | CUDA (T4) | CUDA (T4) |
