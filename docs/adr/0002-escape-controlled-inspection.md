# ADR-0002: Calibrated probabilities and an escape-controlled accept threshold

**Status:** Accepted · 2026-10-12

## Context
Accepting a defective tablet (an escape) is a quality and patient-safety problem; rejecting a good one costs a manual
re-inspection. Arg-max classification treats both errors alike, and raw network probabilities are miscalibrated.

## Decision
Temperature scaling on validation data, then accept only if P(good) ≥ τ, with τ the smallest threshold whose
validation escape rate is ≤ 0.5 % (a margin under the 1.5 % gate on unseen data). Batch statistics of incoming images
are compared with the training reference to detect camera changes; a changed camera gets fine-tuning on a small
labelled set and a new τ.

## Consequences
+ Line 1: 0.84 % escapes at 5.7 % false rejects; calibration error 0.004.
+ A blurred new camera triggers the drift alarm; the system degrades to "send to re-inspection" rather than accepting defects.
− If the optics remove the defect from the pixels (fine cracks under blur), no model setting fixes it; the camera must.
