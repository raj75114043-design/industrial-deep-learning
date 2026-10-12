# ADR-0001: Deep ensemble with Gaussian outputs for dissolution

**Status:** Accepted · 2026-10-12

## Context
A dissolution prediction is only useful for release decisions if it says how sure it is — especially for batches made
under conditions not seen before (a new API supplier, a new site).

## Decision
Five MLPs, each predicting a mean and a variance and trained with a Gaussian negative log-likelihood from different
seeds. The predictive distribution is the mixture: total variance = mean of member variances (noise) + variance of
member means (model uncertainty). Lab transcription errors are screened out first with cross-validated residuals.

## Consequences
+ RMSE 1.64 % against a 1.49 % noise floor; 92.6 % coverage of the 90 % interval.
+ Model uncertainty rises 4.4× for the new supplier (AUROC 0.99) — the API flags these batches as unfamiliar.
− Coverage on the new supplier is only 71 %: uncertainty grows, but not enough to be a guarantee; such batches still need a lab test.
