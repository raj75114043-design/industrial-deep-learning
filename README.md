# Industrial Deep Learning — Tablet Manufacturing

[![CI](https://github.com/raj75114043-design/industrial-deep-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/raj75114043-design/industrial-deep-learning/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://raj-industrial-deep-learning.streamlit.app)
[![GitLab](https://img.shields.io/badge/GitLab-mirror-FC6D26?style=flat-square&logo=gitlab&logoColor=white)](https://gitlab.com/raj75114043/industrial-deep-learning)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ANN_·_CNN_·_LSTM-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-FF3B3B?style=flat-square)

Three PyTorch models for an immediate-release tablet line at a fictional manufacturer, **Meridian Pharma**, each
solving a problem a plant actually has and each evaluated the way a quality or engineering team would ask:

| | Problem | Model | Business question |
|---|---|---|---|
| **ANN** | Dissolution is known only after the lab test, days after compression | Deep ensemble of Gaussian MLPs on process parameters and in-process controls | Will this batch meet Q = 80 %? How sure are we? Which parameter moves it? |
| **CNN** | Manual visual inspection is slow and inconsistent | Convolutional network on camera images, calibrated, with an escape-controlled accept threshold and Grad-CAM | Accept or reject each tablet without letting defects through |
| **RNN** | Punch tooling fails mid-batch, causing weight variation and capping | LSTM on recipe-normalised press sensors with an asymmetric loss | How many cycles are left on this punch set? When do we plan the change? |

> **Live console:** https://raj-industrial-deep-learning.streamlit.app — all batch records, images and sensor logs are synthetic.

---

## The hard parts (and how each model handles them)

**ANN — dissolution prediction**
- *Interactions process scientists know* — coarse API particles dissolve slower; hard tablets dissolve slower; **over-lubrication** (Mg stearate % × blend time) forms a hydrophobic film. The generator encodes these; the console shows the resulting design space.
- *Messy records* — missing in-process values (imputed, with missing-value indicators) and lab transcription errors, found by a cross-validated residual screen and excluded as an OOS investigation would invalidate them.
- *Knowing when it does not know* — a new API supplier with much coarser particles is outside anything seen in training. Each ensemble member predicts a mean and a variance; their disagreement (epistemic uncertainty) rises on unfamiliar batches and the API flags them.

**CNN — visual inspection**
- *Rare, small defects* — chipped, cracked, foreign-particle spot and mottled coating are 28 % of images together; class-weighted loss.
- *Escapes cost more than re-inspection* — a tablet is accepted only if P(good) ≥ τ, with τ chosen on validation for ≤ 0.5 % escapes; probabilities are temperature-scaled first so τ means what it says.
- *A new camera* — line 2 is dimmer, lower-contrast and softer. Without photometric augmentation the network collapses on it; with augmentation it holds most of its accuracy; a drift check on batch statistics raises the alarm; fine-tuning on 600 labelled line-2 images helps further.
- *Explainability* — Grad-CAM shows the region behind each decision.

**RNN — punch remaining useful life**
- *Operating regimes* — three product recipes with different speeds and forces switch every few dozen cycles, so a raw sensor level means nothing; sensors are standardised per recipe using healthy-operation statistics.
- *Asymmetric cost* — predicting more life than remains causes a failure mid-batch; the loss weighs over-estimates ×2 and the NASA score is reported alongside RMSE.
- *Real logs* — sensor dropouts and a drifting temperature probe.

## Results

`python -m indl.eval.run --gate` retrains all three models from scratch (≈ 12 min on 2 CPU cores) and fails CI if any
gate is missed.

**ANN — dissolution (% dissolved at 30 min; 450 held-out batches)**

| Model | RMSE | New API supplier (500 batches) RMSE |
|---|---|---|
| **Deep ensemble (5 × MLP)** | **1.64** | **3.37** |
| Gradient boosting | 2.09 | 15.28 |
| Ridge regression | 4.74 | — |
| *Noise floor (lab repeatability)* | *1.49* | |

| Check | Result | Gate |
|---|---|---|
| R² | 0.95 | |
| 90 % interval coverage | **92.6 %** | ≥ 86 % |
| OOS batches flagged at P(below Q) ≥ 0.2 — recall / precision | **100 % / 74 %** | recall ≥ 95 % |
| Lab-error screen — recall / precision | **7 of 7 / no false flags** | recall ≥ 80 % |
| Model uncertainty on the new supplier vs normal batches | **× 4.4**, separates them with AUROC **0.99** | × 2, AUROC ≥ 0.9 |
| 90 % interval coverage on the new supplier | 70.8 % | reported |

**CNN — inspection (3,000 test images, line 1)**

| Check | Result | Gate |
|---|---|---|
| Accuracy / macro-F1 | **98.5 % / 0.970** | ≥ 97 % / 0.95 |
| F1 per class: good · chipped · cracked · spot · mottled | 0.99 · 1.00 · 0.92 · 0.96 · 0.98 | |
| **Escape rate** (defective tablets accepted) | **0.84 %** | ≤ 1.5 % |
| False reject rate (good tablets sent to re-inspection) | **5.7 %** | ≤ 12 % |
| Expected calibration error, before → after temperature scaling | 0.018 → **0.004** | ≤ 0.03 |
| Line-2 camera accuracy, without → with photometric augmentation | 6.4 % → **92.9 %** | gain ≥ 20 pts |
| Drift alarm: line 2 / line 1 | **raised / not raised** | same |
| Line 2 after fine-tuning on 600 images: accuracy (line 1 after) | **95.6 %** (98.3 %) | gain ≥ 1.5 pts, line 1 ≥ 97 % |

**RNN — punch RUL (100 punch sets in service, last window per unit)**

| Model | RMSE (cycles) | NASA score (lower is better) |
|---|---|---|
| **LSTM, recipe-normalised, asymmetric loss** | **11.0** | **189** |
| GRU, same inputs | 11.7 | 225 |
| LSTM without recipe normalisation | 11.5 | 235 |
| Gradient boosting on window statistics | 12.4 | 263 |

Early warning (alarm when predicted RUL ≤ 23 cycles, threshold set on validation for 90 % recall of "RUL ≤ 25"):
**recall 76 %, precision 68 %** on test windows (gradient boosting: 72 % / 97 %).

### Read these numbers honestly

- **All data is synthetic** — physics-inspired, with planted effects, which is what makes the noise floor and the right answers checkable. Real lines have effects nobody planted; expect lower numbers and re-validate before any GxP use (this would be a GAMP 5 Category 5 system needing formal validation).
- **The ANN beats gradient boosting here** because the response is smooth and the network can extrapolate; on many tabular problems boosting wins — that is why it is in the scorecard. On the new supplier both degrade, the ensemble far less, but its 90 % interval covers only 71 %: rising uncertainty is a warning, not a guarantee.
- **Line-2 cracks stay hard.** Blur removes the thin fissures from the pixels; fine-tuning improves overall accuracy but crack recall on line 2 is ~72–78 %, and holding escapes at ≤ 0.5 % there sends most tablets to re-inspection (false reject 78 %). The right response to the drift alarm is to fix the camera focus, then retrain.
- **The LSTM wins on RMSE and the asymmetric score but its alarms are less precise than gradient boosting's.** It predicts early on purpose (late is costlier); a planner who cares more about unnecessary tool changes would pick the GBM alarm or a lower threshold.
- Gates have margins because PyTorch training on different CPUs is not bit-identical; CI retrains from scratch on every push.

## Quick start

```bash
pip install -r requirements-dev.txt     # CPU-only PyTorch
make test                               # unit + API tests (uses the committed weights in models/)
make models                             # retrain everything, apply the gate, refresh models/ (~12 min)
make console                            # Streamlit console on :8501
make api                                # REST API on :8000 (X-API-Key: dev-key)
```

```bash
curl -s localhost:8000/v1/dissolution -H 'X-API-Key: dev-key' -H 'Content-Type: application/json' \
  -d '{"mgst_pct": 1.4, "lube_blend_min": 9, "compression_force_kn": 18}'
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/dissolution` | predicted dissolution, 90 % interval, P(below Q), "unfamiliar conditions" flag, parameter sensitivity |
| `POST /v1/inspect` | 48 × 48 images → accept / reject + defect type and class probabilities (`camera`: line1 / line2 model) |
| `POST /v1/camera-drift` | ≥ 30 images → brightness / contrast / sharpness z-scores vs training and an alarm |
| `POST /v1/rul` | punch sensor history → RUL, alarm and recommended action |
| `GET /v1/fleet` | RUL for every punch set in service, most urgent first |
| `GET /v1/model` | evaluation card |
| `GET /health`, `/ready`, `/metrics` | probes and Prometheus metrics |

## Repository layout

```
indl/
  data/tablets.py   batch records (+ new-supplier set)        data/images.py   camera images (+ line-2 camera)
  data/sensors.py   run-to-failure punch sensor logs
  common.py         training loop, temperature scaling, ECE
  tabular.py        ANN: preprocessing, outlier screen, Gaussian MLP deep ensemble, sensitivity
  vision.py         CNN: architecture, augmentation, escape-controlled threshold, Grad-CAM, drift, adaptation
  sequence.py       RNN: recipe normalisation, windows, LSTM / GRU, asymmetric loss, baseline, metrics
  service.py        loads production weights for the API and console
  api.py · cli.py · config.py
  eval/run.py       evaluation and quality gate (--save refreshes models/)
models/             production weights (< 1 MB in total) and the fleet snapshot
streamlit_app.py    console: design space, inspection with Grad-CAM, punch fleet
docs/               architecture, evaluation, runbook, ADRs
```

## Design decisions

- [ADR-0001](docs/adr/0001-deep-ensemble-for-dissolution.md) — Deep ensemble with Gaussian outputs for dissolution
- [ADR-0002](docs/adr/0002-escape-controlled-inspection.md) — Calibrated probabilities and an escape-controlled accept threshold
- [ADR-0003](docs/adr/0003-recipe-normalised-lstm.md) — Recipe normalisation and an asymmetric loss for punch RUL
- [ADR-0004](docs/adr/0004-small-cpu-models.md) — Small CPU models with committed weights

## Limitations and roadmap

- Real images would be larger and colour; a pretrained backbone (ResNet-18 / EfficientNet) with the same calibration and threshold logic is the natural next step.
- Dissolution profiles (several time points) instead of a single Q value; Bayesian optimisation over the design space.
- RUL with uncertainty (quantile or ensemble LSTM) so maintenance can be planned against a risk level.
- ONNX export and an edge deployment path for the inspection camera.

## Licence

MIT
