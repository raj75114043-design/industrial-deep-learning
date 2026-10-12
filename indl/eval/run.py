"""Evaluation and quality gate for the three deep-learning models.

    python -m indl.eval.run --gate [--save]

* ANN — dissolution: deep ensemble vs gradient boosting vs ridge; interval coverage; OOS screening; a new API supplier
  (out of distribution): extrapolation error and whether uncertainty rises.
* CNN — visual inspection: accuracy, macro-F1, escapes and false rejects at the calibrated threshold; calibration
  (ECE); line-2 camera shift with and without photometric augmentation; drift alarm; few-shot adaptation.
* RNN — punch RUL: LSTM vs GRU vs LSTM without recipe normalisation vs gradient boosting; RMSE, NASA score and
  early-warning precision / recall at a threshold chosen on validation.

``--save`` writes the production weights to ``models/`` (loaded by the API and the console).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import f1_score, r2_score, roc_auc_score

from indl import sequence as Q
from indl import tabular as T
from indl import vision as V
from indl.common import ece
from indl.data import images, sensors, tablets

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
THRESHOLDS = {  # name: (threshold, "min" | "max")
    "ann_rmse": (2.0, "max"), "ann_beats_gbm": (1, "min"), "ann_beats_ridge": (1, "min"), "ann_coverage90": (0.86, "min"),
    "ann_oos_recall": (0.95, "min"), "screen_recall": (0.8, "min"), "ann_ood_beats_gbm": (1, "min"), "ann_ood_uncertainty_ratio": (2.0, "min"),
    "ann_ood_auroc": (0.9, "min"),
    "cnn_accuracy": (0.97, "min"), "cnn_macro_f1": (0.95, "min"), "cnn_escape_rate": (0.015, "max"), "cnn_false_reject": (0.12, "max"),
    "cnn_ece": (0.03, "max"), "cnn_aug_shift_gain": (0.2, "min"), "cnn_drift_alarm_line2": (1, "min"), "cnn_drift_false_alarm_line1": (0, "max"),
    "cnn_adapted_gain": (0.015, "min"), "cnn_adapted_line1_accuracy": (0.97, "min"),
    "rnn_rmse_last": (13.0, "max"), "rnn_beats_gbm_nasa": (1, "min"), "rnn_recipe_norm_helps": (1, "min"), "rnn_warning_recall": (0.7, "min"),
    "rnn_warning_precision": (0.55, "min"),
}


def ann(m: dict, save: bool) -> None:
    d = tablets.generate()
    ood = tablets.generate(500, ood=True)
    idx = np.random.default_rng(0).permutation(len(d))
    tr, va, te = d.iloc[idx[:2100]], d.iloc[idx[2100:2550]], d.iloc[idx[2550:]]
    flags = T.screen_outliers(tr)
    m["screen_flagged"], m["screen_true_errors"] = int(flags.sum()), int(tr["lab_error"].sum())
    m["screen_recall"] = float((flags & tr["lab_error"]).sum() / max(tr["lab_error"].sum(), 1))
    m["screen_precision"] = float((flags & tr["lab_error"]).sum() / max(flags.sum(), 1))
    trc, vac, tec = tr[~flags], va[~T.screen_outliers(va)], te[~te["lab_error"]]
    t0 = time.time()
    ens = T.train_ensemble(trc, vac)
    m["ann_train_s"] = round(time.time() - t0, 1)
    p, y = ens.predict(tec), tec[T.TARGET].to_numpy()
    gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(trc[T.FEATURES], trc[T.TARGET])
    ridge = RidgeCV().fit(ens.prep.transform(trc).numpy(), trc[T.TARGET])
    pg, pr = gbm.predict(tec[T.FEATURES]), ridge.predict(ens.prep.transform(tec).numpy())
    rm = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))  # noqa: E731
    m["ann_rmse"], m["gbm_rmse"], m["ridge_rmse"] = rm(y, p["pred"]), rm(y, pg), rm(y, pr)
    m["ann_r2"] = float(r2_score(y, p["pred"]))
    m["ann_noise_floor_rmse"] = rm(y, tec["dissolution_true"])
    m["ann_beats_gbm"], m["ann_beats_ridge"] = float(m["ann_rmse"] < m["gbm_rmse"]), float(m["ann_rmse"] < m["ridge_rmse"])
    m["ann_coverage90"] = float(((y >= p["lo90"]) & (y <= p["hi90"])).mean())
    oos = tec["oos"].to_numpy()
    flag = p["p_oos"].to_numpy() >= 0.2
    m["ann_oos_recall"], m["ann_oos_precision"] = float((flag & oos).sum() / max(oos.sum(), 1)), float((flag & oos).sum() / max(flag.sum(), 1))
    po, yo = ens.predict(ood), ood[T.TARGET].to_numpy()
    m["ann_ood_rmse"], m["gbm_ood_rmse"] = rm(yo, po["pred"]), rm(yo, gbm.predict(ood[T.FEATURES]))
    m["ann_ood_beats_gbm"] = float(m["ann_ood_rmse"] < m["gbm_ood_rmse"])
    m["ann_ood_uncertainty_ratio"] = float(po["sd_epistemic"].mean() / p["sd_epistemic"].mean())
    m["ann_ood_auroc"] = float(roc_auc_score(np.r_[np.zeros(len(p)), np.ones(len(po))], np.r_[p["sd_epistemic"], po["sd_epistemic"]]))
    m["ann_ood_coverage90"] = float(((yo >= po["lo90"]) & (yo <= po["hi90"])).mean())
    if save:
        torch.save(ens.state(), MODELS / "dissolution_ensemble.pt")


def cnn(m: dict, save: bool) -> None:
    xtr, ytr = images.generate(4000, 1)
    xva, yva = images.generate(2000, 2)
    xte, yte = images.generate(3000, 3)
    xsh, ysh = images.generate(2000, 4, shift=True)
    t0 = time.time()
    ins = V.train_inspector(xtr, ytr, xva, yva, photometric=True, epochs=30)
    m["cnn_train_s"] = round(time.time() - t0, 1)
    plain = V.train_inspector(xtr, ytr, xva, yva, photometric=False, epochs=12)
    p = ins.probs(xte)
    dec = ins.decide(p)
    m["cnn_accuracy"], m["cnn_macro_f1"] = float((p.argmax(1) == yte).mean()), float(f1_score(yte, p.argmax(1), average="macro"))
    m["cnn_per_class_f1"] = dict(zip(images.CLASSES, f1_score(yte, p.argmax(1), average=None).round(4).tolist(), strict=True))
    m["cnn_escape_rate"] = float(((dec == 0) & (yte != 0)).sum() / (yte != 0).sum())
    m["cnn_false_reject"] = float(((dec != 0) & (yte == 0)).sum() / (yte == 0).sum())
    m["cnn_ece_uncalibrated"], m["cnn_ece"] = ece(ins.probs(xte, calibrated=False), yte), ece(p, yte)
    m["cnn_temperature"], m["cnn_tau_good"] = ins.temperature, ins.tau_good
    m["cnn_params"] = ins.meta["params"]
    m["cnn_shift_accuracy"] = {"with_photometric_aug": float((ins.probs(xsh).argmax(1) == ysh).mean()),
                               "without": float((plain.probs(xsh).argmax(1) == ysh).mean())}
    m["cnn_plain_in_domain_accuracy"] = float((plain.probs(xte).argmax(1) == yte).mean())
    m["cnn_aug_shift_gain"] = m["cnn_shift_accuracy"]["with_photometric_aug"] - m["cnn_shift_accuracy"]["without"]
    dsh = ins.decide(ins.probs(xsh))
    m["cnn_line2_before_adapt"] = {"escape": float(((dsh == 0) & (ysh != 0)).sum() / (ysh != 0).sum()),
                                   "false_reject": float(((dsh != 0) & (ysh == 0)).sum() / (ysh == 0).sum())}
    ref = V.image_stats(xtr)
    m["cnn_drift_line1"], m["cnn_drift_line2"] = V.drift_check(ref, xte[:200]), V.drift_check(ref, xsh[:200])
    m["cnn_drift_alarm_line2"], m["cnn_drift_false_alarm_line1"] = float(m["cnn_drift_line2"]["alarm"]), float(m["cnn_drift_line1"]["alarm"])
    xa, ya = images.generate(600, 5, shift=True)
    xav, yav = images.generate(1000, 6, shift=True)
    adapted = V.adapt(ins, xa, ya, xtr[:600], ytr[:600], xav, yav)
    da = adapted.decide(adapted.probs(xsh))
    m["cnn_adapted_accuracy"] = float((adapted.probs(xsh).argmax(1) == ysh).mean())
    m["cnn_adapted_escape"] = float(((da == 0) & (ysh != 0)).sum() / (ysh != 0).sum())
    m["cnn_adapted_false_reject"] = float(((da != 0) & (ysh == 0)).sum() / (ysh == 0).sum())
    m["cnn_adapted_line1_accuracy"] = float((adapted.probs(xte).argmax(1) == yte).mean())
    m["cnn_adapted_gain"] = m["cnn_adapted_accuracy"] - m["cnn_shift_accuracy"]["with_photometric_aug"]
    crack = ysh == 2
    m["cnn_line2_crack_recall"] = {"base": float((ins.probs(xsh).argmax(1)[crack] == 2).mean()),
                                   "adapted": float((adapted.probs(xsh).argmax(1)[crack] == 2).mean())}
    if save:
        torch.save(ins.state(), MODELS / "inspector.pt")
        torch.save(adapted.state(), MODELS / "inspector_line2.pt")
        np.save(MODELS / "image_reference_stats.npy", ref)


def rnn(m: dict, save: bool) -> None:
    tr, va = sensors.generate(120, 1), sensors.generate(30, 2)
    te_full = sensors.generate(100, 3)
    te = sensors.truncate(te_full, 3)
    results, models = {}, {}
    for name, cell, norm in (("lstm", "lstm", True), ("gru", "gru", True), ("lstm_no_recipe_norm", "lstm", False)):
        t0 = time.time()
        mod = Q.train_rnn(tr, va, cell, norm)
        x, y, _ = Q.windows(te, mod.scaler, last_only=True)
        pr = mod.predict(x)
        results[name] = {"rmse_last": Q.rmse(y, pr), "nasa": Q.nasa_score(y, pr), "train_s": round(time.time() - t0, 1)}
        models[name] = mod
    sc = Q.RecipeScaler().fit(tr)
    gb = Q.train_baseline(tr, sc)
    x, y, _ = Q.windows(te, sc, last_only=True)
    pg = np.clip(gb.predict(Q.summary_features(x)), 0, Q.RUL_CAP)
    results["gbm_window_stats"] = {"rmse_last": Q.rmse(y, pg), "nasa": Q.nasa_score(y, pg)}
    m["rnn_results"] = results
    m["rnn_rmse_last"], m["rnn_nasa"] = results["lstm"]["rmse_last"], results["lstm"]["nasa"]
    m["rnn_beats_gbm_nasa"] = float(results["lstm"]["nasa"] < results["gbm_window_stats"]["nasa"])
    m["rnn_recipe_norm_helps"] = float(results["lstm"]["nasa"] < results["lstm_no_recipe_norm"]["nasa"])
    # early warning: alarm threshold chosen on validation for ≥ 90 % recall of "RUL ≤ 25", then scored on test windows
    best = models["lstm"]
    xv, yv, _ = Q.windows(va, best.scaler)
    pv = best.predict(xv)
    thr = next((t for t in np.arange(10, 60, 1.0) if Q.warning_metrics(yv, pv, t)["recall"] >= 0.9), 40.0)   # alarm threshold
    xa, ya, _ = Q.windows(te, best.scaler)
    pa = best.predict(xa)
    w = Q.warning_metrics(ya, pa, thr)
    m["rnn_warning_threshold"] = float(thr)
    m["rnn_warning_recall"], m["rnn_warning_precision"], m["rnn_warning_alarms"] = w["recall"], w["precision"], w["alarms"]
    pgw = np.clip(gb.predict(Q.summary_features(Q.windows(te, sc)[0])), 0, Q.RUL_CAP)
    pgv = np.clip(gb.predict(Q.summary_features(Q.windows(va, sc)[0])), 0, Q.RUL_CAP)
    yvg = Q.windows(va, sc)[1]
    thr_g = next((t for t in np.arange(10, 60, 1.0) if Q.warning_metrics(yvg, pgv, t)["recall"] >= 0.9), 40.0)
    m["gbm_warning"] = Q.warning_metrics(Q.windows(te, sc)[1], pgw, thr_g)
    if save:
        torch.save(best.state(), MODELS / "rul_lstm.pt")
        te.to_csv(MODELS / "fleet_snapshot.csv.gz", index=False)
        (MODELS / "rul_warning.json").write_text(json.dumps({"threshold": float(thr)}))


def evaluate(save: bool = False, parts: tuple = ("ann", "rnn", "cnn"), base: dict | None = None) -> dict:
    MODELS.mkdir(exist_ok=True)
    m: dict = {**(base or {}), "torch": torch.__version__}
    for part in (f for f in (ann, rnn, cnn) if f.__name__ in parts):
        t0 = time.time()
        part(m, save)
        m[f"{part.__name__}_section_s"] = round(time.time() - t0, 1)
        logging.getLogger("indl.eval").warning("%s done in %.0fs", part.__name__, time.time() - t0)
    return m


def gate(m: dict) -> list[str]:
    out = []
    for k, (thr, kind) in THRESHOLDS.items():
        v = m[k]
        if (kind == "min" and v < thr) or (kind == "max" and v > thr):
            out.append(f"{k}={v:.4f} (gate {'>=' if kind == 'min' else '<='} {thr})")
    return out


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--save", action="store_true", help="write production weights to models/")
    ap.add_argument("--out", default=str(ROOT / "reports" / "eval.json"))
    ap.add_argument("--parts", default="ann,rnn,cnn", help="re-run only some sections, merging into the existing report")
    a = ap.parse_args()
    parts = tuple(a.parts.split(","))
    base = json.loads(Path(a.out).read_text()) if len(parts) < 3 and Path(a.out).exists() else None
    m = evaluate(a.save, parts, base)
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(m, indent=2, default=str))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, indent=1, default=str))
    if a.save:
        (MODELS / "card.json").write_text(json.dumps({k: m[k] for k in m if not isinstance(m[k], dict)}, indent=2, default=str))
    br = gate(m)
    if a.gate:
        print("QUALITY GATE " + ("FAILED: " + "; ".join(br) if br else "PASSED"))
        sys.exit(1 if br else 0)


if __name__ == "__main__":
    main()
