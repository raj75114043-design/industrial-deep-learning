"""Serving layer: loads the production weights from ``models/`` and exposes the three models to the API and console."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from indl import sequence as Q
from indl import tabular as T
from indl import vision as V
from indl.data.images import CLASSES
from indl.data.tablets import PARAMS, SPEC_Q

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


@dataclass
class Bundle:
    ensemble: T.DissolutionEnsemble
    inspectors: dict
    image_ref: np.ndarray
    rul: Q.RULModel
    rul_threshold: float
    fleet: pd.DataFrame
    card: dict


def _load(name: str):
    return torch.load(MODELS / name, weights_only=False)


@lru_cache(maxsize=1)
def bundle() -> Bundle:
    if not (MODELS / "inspector.pt").exists():
        raise FileNotFoundError("models/ is empty — run `python -m indl.eval.run --save` first")
    return Bundle(
        ensemble=T.DissolutionEnsemble.from_state(_load("dissolution_ensemble.pt")),
        inspectors={"line1": V.Inspector.from_state(_load("inspector.pt")), "line2": V.Inspector.from_state(_load("inspector_line2.pt"))},
        image_ref=np.load(MODELS / "image_reference_stats.npy"),
        rul=Q.RULModel.from_state(_load("rul_lstm.pt")),
        rul_threshold=json.loads((MODELS / "rul_warning.json").read_text())["threshold"],
        fleet=pd.read_csv(MODELS / "fleet_snapshot.csv.gz"),
        card=json.loads((MODELS / "card.json").read_text()) if (MODELS / "card.json").exists() else {})


def predict_dissolution(batch: dict) -> dict:
    d = pd.DataFrame([batch])
    for c in T.FEATURES:
        if c not in d:
            d[c] = np.nan
    p = bundle().ensemble.predict(d).iloc[0]
    sens = T.sensitivity(bundle().ensemble, d.fillna(bundle().ensemble.prep.median))
    return {"dissolution_pct": round(float(p["pred"]), 2), "sd": round(float(p["sd"]), 2), "interval90": [round(float(p["lo90"]), 1), round(float(p["hi90"]), 1)],
            "p_below_spec": round(float(p["p_oos"]), 4), "spec_q": SPEC_Q, "model_uncertainty_sd": round(float(p["sd_epistemic"]), 2),
            "unfamiliar_conditions": bool(p["sd_epistemic"] > 1.0),
            "sensitivity_per_5pct_increase": {k: round(v, 2) for k, v in sens.items()}}


def inspect(images: np.ndarray, camera: str = "line1") -> list[dict]:
    ins = bundle().inspectors[camera]
    p = ins.probs(images)
    dec = ins.decide(p)
    return [{"decision": "accept" if d == 0 else "reject", "defect": None if d == 0 else CLASSES[d], "p_good": round(float(pi[0]), 4),
             "probabilities": {c: round(float(v), 4) for c, v in zip(CLASSES, pi, strict=True)}} for d, pi in zip(dec, p, strict=True)]


def camera_drift(images: np.ndarray) -> dict:
    return V.drift_check(bundle().image_ref, images)


def predict_rul(history: pd.DataFrame) -> dict:
    b = bundle()
    h = history.copy()
    if "unit" not in h:
        h["unit"] = 0
    x, _, _ = Q.windows(h.drop(columns=["rul"], errors="ignore"), b.rul.scaler, last_only=True)
    r = float(b.rul.predict(x)[0])
    return {"rul_cycles": round(r, 1), "alarm": r <= b.rul_threshold, "alarm_threshold": b.rul_threshold,
            "action": "plan punch change at next changeover" if r <= b.rul_threshold else "no action"}


def fleet_status() -> pd.DataFrame:
    b = bundle()
    rows = []
    for uid, g in b.fleet.groupby("unit"):
        r = predict_rul(g.drop(columns=["rul", "life"], errors="ignore"))
        rows.append({"unit": int(uid), "cycles_run": int(g["cycle"].max()), "recipe_now": g["recipe"].iloc[-1], **r,
                     "true_rul": int(g["rul"].iloc[-1]) if "rul" in g else None})
    return pd.DataFrame(rows).sort_values("rul_cycles").reset_index(drop=True)


DEFAULT_BATCH = {"site": "Goa", "api_d50_um": 44.0, "granulation_water_pct": 11.0, "granulation_time_min": 5.5, "impeller_rpm": 220.0, "drying_lod_pct": 2.3,
                 "mgst_pct": 1.0, "lube_blend_min": 5.0, "compression_force_kn": 14.0, "turret_rpm": 40.0, "coating_gain_pct": 3.0, "inlet_temp_c": 62.0,
                 "hardness_kp": 9.7, "friability_pct": 0.3, "thickness_mm": 4.04}
__all__ = ["PARAMS", "DEFAULT_BATCH", "bundle", "predict_dissolution", "inspect", "camera_drift", "predict_rul", "fleet_status"]
