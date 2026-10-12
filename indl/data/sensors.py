"""Run-to-failure sensor histories of rotary tablet-press punch sets (fictional Meridian Pharma, Goa site).

Each unit is one set of upper punches, logged once per production cycle (≈ 1 hour) until it fails (tip wear or
cracking that causes weight variation and capping). Six sensors depend on the punch's hidden health *and* on which
product recipe is running — three recipes with different speeds and forces switch every few dozen cycles, so a raw
sensor level means nothing without the recipe (the same difficulty as NASA C-MAPSS FD002). Degradation starts at a
random point and accelerates. Real-log defects: sensor dropouts (missing values) and a slow temperature-probe drift.
The label is remaining useful life (cycles to failure), capped at 125 as is usual: early in life RUL is unknowable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SENSORS = ["main_force_kn", "force_sd_kn", "ejection_force_n", "vibration_rms_g", "punch_temp_c", "weight_rsd_pct"]
RECIPES = {  # turret rpm, main force, ejection, temperature offset
    "R1-paracetamol-500": (60, 18.0, 210, 0.0), "R2-metformin-850": (45, 24.0, 300, 2.5), "R3-atorvastatin-20": (75, 12.0, 150, -1.5)}
RUL_CAP = 125


def unit(r: np.random.Generator, uid: int) -> pd.DataFrame:
    life = int(r.integers(160, 340))
    onset = life - int(r.integers(70, 140))
    t = np.arange(1, life + 1)
    h = np.clip((t - onset) / (life - onset), 0, 1) ** r.uniform(1.4, 2.2)   # hidden wear 0 → 1 at failure
    names = list(RECIPES)
    recipe, cur = [], r.integers(3)
    while len(recipe) < life:
        recipe += [names[cur]] * int(r.integers(10, 45))
        cur = (cur + r.integers(1, 3)) % 3
    recipe = np.array(recipe[:life])
    rpm, force, ej, toff = (np.array([RECIPES[x][i] for x in recipe]) for i in range(4))
    drift = np.cumsum(r.normal(0, 0.01, life)) if r.random() < 0.3 else 0.0
    d = pd.DataFrame({
        "unit": uid, "cycle": t, "recipe": recipe, "turret_rpm": rpm + r.normal(0, 0.5, life),
        "main_force_kn": force * (1 + 0.04 * h) + r.normal(0, 0.25, life),
        "force_sd_kn": 0.3 + 0.004 * rpm + 0.9 * h ** 1.5 + r.normal(0, 0.05, life),
        "ejection_force_n": ej * (1 + 0.35 * h) + r.normal(0, 6, life),
        "vibration_rms_g": 0.05 + 0.002 * rpm + 0.25 * h * rpm / 60 + r.normal(0, 0.015, life),
        "punch_temp_c": 31 + toff + 0.03 * rpm + 6 * h + drift + r.normal(0, 0.4, life),
        "weight_rsd_pct": 0.6 + 1.6 * h ** 2 + r.normal(0, 0.08, life)})
    for c in SENSORS:
        d.loc[r.random(life) < 0.02, c] = np.nan
    d["rul"] = np.minimum(life - t, RUL_CAP)
    d["life"] = life
    return d


def generate(n_units: int, seed: int) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    return pd.concat([unit(r, i + 1) for i in range(n_units)], ignore_index=True)


def truncate(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Test units are observed only up to a random point before failure (like a fleet in service today)."""
    r = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby("unit"):
        cut = int(r.integers(int(0.45 * len(g)), len(g) - 3))
        parts.append(g.iloc[:cut])
    return pd.concat(parts, ignore_index=True)
