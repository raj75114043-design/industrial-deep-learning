# ADR-0003: Recipe normalisation and an asymmetric loss for punch RUL

**Status:** Accepted · 2026-10-12

## Context
Sensor levels depend on the product recipe running (speed, force), which changes every few dozen cycles; wear is a
small change on top. Over-estimating remaining life causes a failure mid-batch; under-estimating wastes some tooling life.

## Decision
Standardise each sensor per recipe with statistics from healthy operation; feed 40-cycle windows plus a missing-value
channel to a 2-layer LSTM; train with a Huber loss that weighs over-estimates twice; report the NASA asymmetric score
and choose the alarm threshold on validation for 90 % recall.

## Consequences
+ NASA score 189 vs 235 without recipe normalisation and 263 for gradient boosting on window statistics.
− Alarms are less precise (68 %) than the boosting baseline's (97 %) at similar recall; the threshold is a maintenance-policy choice.
