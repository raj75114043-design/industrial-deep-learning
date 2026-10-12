"""RNN for remaining-useful-life (RUL) estimation from multivariate sensor windows.

* Recipe normalisation — each sensor is standardised **per recipe** using training statistics, which removes the
  operating-condition effect and leaves the wear signal (without it the network must learn recipes from scratch).
* Missing values — forward-filled within a unit, with the fill rate as an input channel.
* Model — 2-layer LSTM (or GRU) over 40-cycle windows, last hidden state → RUL. The loss is an asymmetric Huber on
  RUL / 125 that weighs over-estimates (predicting more life than is left) twice as much as under-estimates.
* Baseline — gradient boosting on window summary statistics (mean, last, slope, min, max per sensor).
* Metrics — RMSE, the asymmetric NASA score (late predictions are penalised more than early ones, because a missed
  failure costs more than an early tool change) and early-warning precision / recall for "RUL ≤ 25 cycles".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch import nn

from indl.common import fit, seed_everything
from indl.data.sensors import RECIPES, RUL_CAP, SENSORS

WINDOW = 40
WARN = 25


@dataclass
class RecipeScaler:
    stats: dict = field(default_factory=dict)
    per_recipe: bool = True

    def fit(self, df: pd.DataFrame) -> RecipeScaler:
        healthy = df[df["rul"] >= RUL_CAP] if "rul" in df else df   # reference = healthy operation
        for rec in RECIPES:
            g = healthy[healthy["recipe"] == rec] if self.per_recipe else healthy
            self.stats[rec] = (g[SENSORS].mean().to_numpy(), g[SENSORS].std().to_numpy() + 1e-6)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[SENSORS].to_numpy(float).copy()
        for rec, (m, s) in self.stats.items():
            k = (df["recipe"] == rec).to_numpy()
            x[k] = (x[k] - m) / s
        return x


def windows(df: pd.DataFrame, scaler: RecipeScaler, last_only: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ids = [], [], []
    for uid, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        miss = g[SENSORS].isna().mean(axis=1).to_numpy()[:, None]
        z = pd.DataFrame(scaler.transform(g)).ffill().bfill().fillna(0).to_numpy()
        z = np.hstack([z, miss])
        if len(z) < WINDOW:   # left-pad short histories with their first row
            z = np.vstack([np.repeat(z[:1], WINDOW - len(z), 0), z])
        y = np.concatenate([np.full(max(0, WINDOW - len(g)), g["rul"].iloc[0]), g["rul"].to_numpy()]) if "rul" in g else np.zeros(len(z))
        ends = [len(z)] if last_only else range(WINDOW, len(z) + 1)
        for e in ends:
            xs.append(z[e - WINDOW:e])
            ys.append(y[e - 1])
            ids.append(uid)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(ids)


class RULNet(nn.Module):
    def __init__(self, n_in: int = len(SENSORS) + 1, hidden: int = 48, cell: str = "lstm"):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_in, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        self.cell = cell

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(1)


@dataclass
class RULModel:
    net: RULNet
    scaler: RecipeScaler
    meta: dict = field(default_factory=dict)

    def predict(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return np.clip(self.net(torch.tensor(x)).numpy() * RUL_CAP, 0, RUL_CAP)

    def state(self) -> dict:
        return {"net": self.net.state_dict(), "cell": self.net.cell, "stats": {k: [v[0].tolist(), v[1].tolist()] for k, v in self.scaler.stats.items()},
                "meta": self.meta}

    @classmethod
    def from_state(cls, s: dict) -> RULModel:
        net = RULNet(cell=s["cell"])
        net.load_state_dict(s["net"])
        net.eval()
        sc = RecipeScaler({k: (np.array(v[0]), np.array(v[1])) for k, v in s["stats"].items()})
        return cls(net, sc, s.get("meta", {}))


def asymmetric_huber(pred, y, delta: float = 0.1, late_weight: float = 2.0):
    d = pred - y
    a = d.abs()
    h = torch.where(a < delta, 0.5 * d ** 2, delta * (a - 0.5 * delta))
    return (h * torch.where(d > 0, late_weight, 1.0)).mean()


def train_rnn(train_df: pd.DataFrame, val_df: pd.DataFrame, cell: str = "lstm", per_recipe: bool = True, epochs: int = 40, seed: int = 0) -> RULModel:
    seed_everything(seed)
    sc = RecipeScaler(per_recipe=per_recipe).fit(train_df)
    xt, yt, _ = windows(train_df, sc)
    xv, yv, _ = windows(val_df, sc)
    net = RULNet(cell=cell)
    h = fit(net, (torch.tensor(xt), torch.tensor(yt / RUL_CAP)), (torch.tensor(xv), torch.tensor(yv / RUL_CAP)), asymmetric_huber,
            epochs=epochs, lr=2e-3, batch=256, patience=8, seed=seed)
    return RULModel(net, sc, {"cell": cell, "per_recipe": per_recipe, "epochs_run": h["epochs_run"], "train_windows": len(yt)})


def summary_features(x: np.ndarray) -> np.ndarray:
    t = np.arange(WINDOW) - (WINDOW - 1) / 2
    slope = (x * t[None, :, None]).sum(1) / (t ** 2).sum()
    return np.hstack([x.mean(1), x[:, -5:].mean(1), slope, x.min(1), x.max(1)])


def train_baseline(train_df: pd.DataFrame, scaler: RecipeScaler) -> HistGradientBoostingRegressor:
    x, y, _ = windows(train_df, scaler)
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(summary_features(x), y)


def nasa_score(y: np.ndarray, p: np.ndarray) -> float:
    d = p - y
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def warning_metrics(y: np.ndarray, p: np.ndarray, alarm_at: float = WARN, warn: int = WARN) -> dict:
    """Alarm when predicted RUL ≤ alarm_at; an event is a window whose true RUL ≤ warn."""
    alarm, truth = p <= alarm_at, y <= warn
    tp = (alarm & truth).sum()
    return {"precision": float(tp / max(alarm.sum(), 1)), "recall": float(tp / max(truth.sum(), 1)), "alarms": int(alarm.sum()), "true_events": int(truth.sum())}
