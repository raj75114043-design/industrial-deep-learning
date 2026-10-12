"""Synthetic batch records for an immediate-release tablet (fictional Meridian Pharma, three sites).

Each batch has granulation, drying, blending, compression and coating parameters plus in-process controls, and a
dissolution result (% released at 30 min; specification Q = 80 %). The response contains the interactions process
scientists know: larger API particles dissolve slower, hard tablets dissolve slower, and **over-lubrication**
(magnesium stearate % × blend time) forms a hydrophobic film. Real-record defects are included: missing in-process
values, a few lab transcription errors, and a site offset. An out-of-distribution set uses a new API supplier with
much coarser particles — conditions the model has never seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SITES = ["Goa", "Baddi", "Sikkim"]
PARAMS = ["api_d50_um", "granulation_water_pct", "granulation_time_min", "impeller_rpm", "drying_lod_pct", "mgst_pct", "lube_blend_min",
          "compression_force_kn", "turret_rpm", "coating_gain_pct", "inlet_temp_c"]
IPC = ["hardness_kp", "friability_pct", "thickness_mm"]
SPEC_Q = 80.0


def _true_dissolution(d: pd.DataFrame) -> np.ndarray:
    lube = d["mgst_pct"] * d["lube_blend_min"]
    return (101.0
            - 0.30 * (d["api_d50_um"] - 40) - 0.006 * np.clip(d["api_d50_um"] - 40, 0, None) ** 2
            - 2.0 * np.clip(d["hardness_kp"] - 10, 0, None) ** 1.3
            - 5.0 * np.clip(lube - 7, 0, None) ** 0.85
            - 0.9 * (d["granulation_water_pct"] - 11) ** 2
            - 2.5 * np.clip(1.8 - d["drying_lod_pct"], 0, None)
            - 3.0 * (d["coating_gain_pct"] - 3)
            - np.where(d["site"] == "Sikkim", 2.0, 0.0))


def generate(n: int = 3000, seed: int = 11, ood: bool = False) -> pd.DataFrame:
    r = np.random.default_rng(seed + (1000 if ood else 0))
    site = r.choice(SITES, n, p=[0.45, 0.35, 0.2])
    d = pd.DataFrame({"batch": [f"{'OOD' if ood else 'B'}{i:05d}" for i in range(n)], "site": site})
    d["api_d50_um"] = r.normal(82, 6, n) if ood else r.normal(44, 7, n)
    d["granulation_water_pct"] = r.normal(11, 1.0, n)
    d["granulation_time_min"] = r.uniform(3, 8, n)
    d["impeller_rpm"] = r.uniform(150, 300, n)
    d["drying_lod_pct"] = np.clip(r.normal(2.3, 0.5, n), 0.6, 4.5)
    d["mgst_pct"] = r.uniform(0.5, 1.5, n)
    d["lube_blend_min"] = r.uniform(2, 10, n)
    d["compression_force_kn"] = r.uniform(8, 20, n) + np.where(site == "Baddi", 1.0, 0.0)
    d["turret_rpm"] = r.uniform(20, 60, n)
    d["coating_gain_pct"] = r.normal(3, 0.4, n)
    d["inlet_temp_c"] = r.uniform(55, 70, n)
    gran = (d["granulation_water_pct"] - 11) * 0.3 + (d["granulation_time_min"] - 5.5) * 0.15
    d["hardness_kp"] = 2 + 0.55 * d["compression_force_kn"] + 0.8 * gran - 0.6 * (d["drying_lod_pct"] - 2.3) - 0.02 * (d["turret_rpm"] - 40) + r.normal(0, 0.5, n)
    d["friability_pct"] = np.clip(0.9 - 0.06 * d["hardness_kp"] + r.normal(0, 0.05, n), 0.05, None)
    d["thickness_mm"] = 4.6 - 0.04 * d["compression_force_kn"] + r.normal(0, 0.03, n)
    d["dissolution_true"] = np.clip(_true_dissolution(d), 20, 102)
    d["dissolution_pct"] = np.clip(d["dissolution_true"] + r.normal(0, 1.5, n), 0, 105).round(1)
    # record defects: missing IPC values, lab transcription errors
    for c, p in (("drying_lod_pct", 0.04), ("hardness_kp", 0.03), ("friability_pct", 0.05)):
        d.loc[r.random(n) < p, c] = np.nan
    if not ood:
        bad = r.random(n) < 0.004
        d.loc[bad, "dissolution_pct"] = r.uniform(5, 40, bad.sum()).round(1)
        d["lab_error"] = bad
    else:
        d["lab_error"] = False
    d["oos"] = d["dissolution_pct"] < SPEC_Q
    return d
