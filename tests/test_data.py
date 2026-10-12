import numpy as np

from indl.data import images, sensors, tablets


def test_tablet_records_have_known_defects():
    d = tablets.generate(1000)
    assert d["drying_lod_pct"].isna().any() and d["lab_error"].any()
    assert 0.03 < d["oos"].mean() < 0.15
    assert tablets.generate(200, ood=True)["api_d50_um"].mean() > 70


def test_overlubrication_lowers_dissolution():
    d = tablets.generate(1, seed=1).iloc[[0]].copy()
    d["hardness_kp"], d["site"] = 9.0, "Goa"
    lo = tablets._true_dissolution(d.assign(mgst_pct=0.6, lube_blend_min=3))
    hi = tablets._true_dissolution(d.assign(mgst_pct=1.5, lube_blend_min=10))
    assert float(lo.iloc[0]) - float(hi.iloc[0]) > 10


def test_images_shape_classes_and_shift():
    x, y = images.generate(50, 0, balanced=True)
    xs, _ = images.generate(50, 0, shift=True, balanced=True)
    assert x.shape == (50, 48, 48) and x.dtype == np.float32 and set(y) <= set(range(5))
    assert xs.std() < x.std()   # lower contrast camera


def test_sensor_units_run_to_failure():
    d = sensors.generate(5, 0)
    for _, g in d.groupby("unit"):
        assert g["rul"].iloc[-1] == 0 and g["rul"].max() <= sensors.RUL_CAP
    t = sensors.truncate(d, 0)
    assert (t.groupby("unit")["rul"].last() > 0).all()
