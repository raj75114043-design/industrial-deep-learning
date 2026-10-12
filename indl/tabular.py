"""ANN for tabular process data: dissolution prediction with a deep ensemble.

* Pre-processing — median imputation with missing-value indicators, one-hot site, standardisation (fitted on train only).
* Outlier screening — cross-validated gradient-boosting residuals flag likely lab transcription errors (> 6 robust SD)
  for investigation; flagged batches are excluded from training, as an OOS investigation would invalidate them.
* Model — 5 MLPs (64-64, SiLU, dropout) trained with a Gaussian negative log-likelihood, each predicting a mean and a
  variance. The ensemble mixture gives aleatoric (noise) + epistemic (model disagreement) uncertainty, so batches unlike
  the training data get wider intervals.
* Outputs — mean, standard deviation, 90 % interval and P(dissolution < Q) per batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from torch import nn

from indl.common import fit, seed_everything
from indl.data.tablets import IPC, PARAMS, SITES, SPEC_Q

FEATURES = PARAMS + IPC
TARGET = "dissolution_pct"


@dataclass
class Prep:
    median: dict
    mean: np.ndarray
    std: np.ndarray
    columns: list = field(default_factory=list)

    @classmethod
    def fit(cls, d: pd.DataFrame) -> Prep:
        med = d[FEATURES].median().to_dict()
        x = cls._raw(d, med)
        return cls(med, x.mean(0), x.std(0) + 1e-6, cls._names())

    @staticmethod
    def _names() -> list:
        return FEATURES + [f"{c}_missing" for c in ("drying_lod_pct", "hardness_kp", "friability_pct")] + [f"site_{s}" for s in SITES]

    @staticmethod
    def _raw(d: pd.DataFrame, med: dict) -> np.ndarray:
        x = d[FEATURES].fillna(med).to_numpy(float)
        miss = d[["drying_lod_pct", "hardness_kp", "friability_pct"]].isna().to_numpy(float)
        site = np.stack([(d["site"] == s).to_numpy(float) for s in SITES], 1)
        return np.hstack([x, miss, site])

    def transform(self, d: pd.DataFrame) -> torch.Tensor:
        return torch.tensor((self._raw(d, self.median) - self.mean) / self.std, dtype=torch.float32)


class GaussianMLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout))
        self.mu, self.log_var = nn.Linear(hidden, 1), nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return torch.cat([self.mu(h), self.log_var(h).clamp(-6, 6)], 1)


def gaussian_nll(out, y):
    mu, lv = out[:, 0], out[:, 1]
    return (0.5 * (lv + (y - mu) ** 2 / lv.exp())).mean()


Y_MEAN, Y_SCALE = 90.0, 10.0   # target scaling for stable training


@dataclass
class DissolutionEnsemble:
    prep: Prep
    members: list
    history: list = field(default_factory=list)

    def predict(self, d: pd.DataFrame) -> pd.DataFrame:
        x = self.prep.transform(d)
        with torch.no_grad():
            outs = torch.stack([m(x) for m in self.members])          # members × n × 2
        mus = outs[..., 0].numpy() * Y_SCALE + Y_MEAN
        var = outs[..., 1].exp().numpy() * Y_SCALE ** 2
        mean = mus.mean(0)
        aleatoric, epistemic = var.mean(0), mus.var(0)
        sd = np.sqrt(aleatoric + epistemic)
        return pd.DataFrame({"pred": mean, "sd": sd, "sd_epistemic": np.sqrt(epistemic), "lo90": mean - 1.645 * sd, "hi90": mean + 1.645 * sd,
                             "p_oos": norm.cdf((SPEC_Q - mean) / sd)}, index=d.index)

    def state(self) -> dict:
        return {"prep": self.prep.__dict__, "members": [m.state_dict() for m in self.members]}

    @classmethod
    def from_state(cls, s: dict) -> DissolutionEnsemble:
        prep = Prep(**s["prep"])
        members = []
        for sd in s["members"]:
            m = GaussianMLP(len(prep.columns))
            m.load_state_dict(sd)
            m.eval()
            members.append(m)
        return cls(prep, members)


def screen_outliers(d: pd.DataFrame, k: float = 6.0, seed: int = 0) -> pd.Series:
    """Cross-validated residual screen: |residual| > k × 1.4826 × MAD → suspected lab error."""
    res = np.zeros(len(d))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(d):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed).fit(d.iloc[tr][FEATURES], d.iloc[tr][TARGET])
        res[te] = d.iloc[te][TARGET].to_numpy() - m.predict(d.iloc[te][FEATURES])
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    return pd.Series(np.abs(res) > k * mad, index=d.index)


def train_ensemble(train: pd.DataFrame, val: pd.DataFrame, n_members: int = 5, epochs: int = 150, seed: int = 0) -> DissolutionEnsemble:
    prep = Prep.fit(train)
    xt, xv = prep.transform(train), prep.transform(val)
    yt = torch.tensor((train[TARGET].to_numpy() - Y_MEAN) / Y_SCALE, dtype=torch.float32)
    yv = torch.tensor((val[TARGET].to_numpy() - Y_MEAN) / Y_SCALE, dtype=torch.float32)
    members, hist = [], []
    for i in range(n_members):
        seed_everything(seed + i)
        m = GaussianMLP(xt.shape[1])
        h = fit(m, (xt, yt), (xv, yv), gaussian_nll, epochs=epochs, lr=3e-3, batch=64, patience=20, seed=seed + i)
        members.append(m)
        hist.append(h)
    return DissolutionEnsemble(prep, members, hist)


def sensitivity(ens: DissolutionEnsemble, batch: pd.DataFrame, rel: float = 0.05) -> pd.Series:
    """Change in predicted dissolution for a +5 % change of each process parameter (local what-if)."""
    base = float(ens.predict(batch)["pred"].iloc[0])
    out = {}
    for c in PARAMS + ["hardness_kp"]:
        b = batch.copy()
        b[c] = b[c] * (1 + rel)
        out[c] = float(ens.predict(b)["pred"].iloc[0]) - base
    return pd.Series(out).sort_values()
