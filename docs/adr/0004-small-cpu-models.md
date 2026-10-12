# ADR-0004: Small CPU models with committed weights

**Status:** Accepted · 2026-10-12

## Context
The demo, the API and CI must run on free CPU-only machines; plant edge devices are often CPU-only too.

## Decision
Keep the models small (CNN 41 k parameters, LSTM ≈ 30 k, each MLP ≈ 5 k), install CPU-only PyTorch, commit the
production weights (< 1 MB) so the console and API start instantly, and retrain from scratch in CI on every push to
prove the weights are reproducible from code.

## Consequences
+ Full retraining and evaluation in about 12 minutes on two cores; inference in milliseconds.
− Larger, pretrained vision backbones are out of scope for the free tier; the evaluation logic is backbone-agnostic.
