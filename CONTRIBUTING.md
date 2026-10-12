# Contributing

1. `pip install -r requirements-dev.txt` (CPU PyTorch).
2. `make lint test` for every change; `make models` when model code or data generators change — commit the refreshed `models/` and `reports/eval.json` together.
3. Never lower a gate threshold to make a change pass; explain metric changes in the PR.
4. Escapes (defects accepted) and late RUL predictions are high-severity regressions.
