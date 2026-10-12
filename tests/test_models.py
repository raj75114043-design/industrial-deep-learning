import numpy as np
import pandas as pd
import torch

from indl import sequence as Q
from indl import tabular as T
from indl import vision as V
from indl.common import ece, fit_temperature
from indl.data import sensors, tablets


def test_ensemble_trains_and_reports_uncertainty():
    d = tablets.generate(400, seed=3)
    d = d[~d["lab_error"]]
    ens = T.train_ensemble(d.iloc[:300], d.iloc[300:], n_members=2, epochs=15)
    p = ens.predict(d.iloc[300:])
    assert {"pred", "sd", "lo90", "hi90", "p_oos"} <= set(p.columns)
    assert (p["hi90"] > p["lo90"]).all() and p["p_oos"].between(0, 1).all()
    restored = T.DissolutionEnsemble.from_state(ens.state())
    assert np.allclose(restored.predict(d.iloc[300:])["pred"], p["pred"], atol=1e-4)


def test_outlier_screen_finds_gross_errors():
    d = tablets.generate(800, seed=4)
    flags = T.screen_outliers(d)
    assert (flags & d["lab_error"]).sum() >= d["lab_error"].sum() - 1


def test_cnn_forward_gradcam_and_threshold():
    ins = V.Inspector(V.TabletCNN().eval())
    x = np.random.default_rng(0).random((4, 48, 48)).astype(np.float32)
    p = ins.probs(x)
    assert p.shape == (4, 5) and np.allclose(p.sum(1), 1, atol=1e-5)
    cam, prob = ins.gradcam(x[0])
    assert cam.shape == (48, 48) and 0 <= cam.min() and cam.max() <= 1.0001
    p_val = np.array([[0.99, 0.01, 0, 0, 0], [0.7, 0.3, 0, 0, 0], [0.2, 0.8, 0, 0, 0]])
    assert V.choose_tau(p_val, np.array([0, 1, 1])) > 0.7   # must not accept the defect at 0.7


def test_drift_check():
    r = np.random.default_rng(0)
    ref = V.image_stats(r.random((300, 48, 48)).astype(np.float32))
    assert not V.drift_check(ref, r.random((100, 48, 48)).astype(np.float32))["alarm"]
    assert V.drift_check(ref, (r.random((100, 48, 48)) * 0.5).astype(np.float32))["alarm"]


def test_temperature_scaling_reduces_overconfidence():
    g = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 3, (2000,), generator=g)
    logits = torch.randn(2000, 3, generator=g) * 0.5
    logits[torch.arange(2000), labels] += 1.0
    logits *= 4
    flip = torch.rand(2000, generator=g) < 0.3   # 30 % of labels disagree with confident logits → overconfident
    labels = torch.where(flip, torch.randint(0, 3, (2000,), generator=g), labels)
    t = fit_temperature(logits, labels)
    assert t > 1.5
    raw, cal = (torch.softmax(logits / s, 1).numpy() for s in (1.0, t))
    assert ece(cal, labels.numpy()) < ece(raw, labels.numpy())


def test_rul_windows_and_metrics():
    d = sensors.generate(3, 1)
    sc = Q.RecipeScaler().fit(d)
    x, y, ids = Q.windows(d, sc)
    assert x.shape[1:] == (Q.WINDOW, len(sensors.SENSORS) + 1) and len(x) == len(y) == len(ids)
    xl, _, _ = Q.windows(sensors.truncate(d, 1), sc, last_only=True)
    assert len(xl) == 3
    assert Q.nasa_score(np.array([50.0]), np.array([60.0])) > Q.nasa_score(np.array([50.0]), np.array([40.0]))   # late is worse
    assert Q.warning_metrics(np.array([10, 50]), np.array([12, 60]))["recall"] == 1.0
    assert isinstance(pd.DataFrame(x[0]), pd.DataFrame)
