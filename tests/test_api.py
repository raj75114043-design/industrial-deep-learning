import pytest
from fastapi.testclient import TestClient

from indl.service import MODELS

pytestmark = pytest.mark.skipif(not (MODELS / "inspector.pt").exists(), reason="production weights not built")


@pytest.fixture(scope="module")
def client():
    from indl.api import app
    return TestClient(app)


H = {"X-API-Key": "test-key"}


def test_auth_and_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/v1/dissolution", json={}).status_code == 401


def test_dissolution_overlubrication(client):
    ok = client.post("/v1/dissolution", headers=H, json={"mgst_pct": 0.6, "lube_blend_min": 3}).json()
    bad = client.post("/v1/dissolution", headers=H, json={"mgst_pct": 1.5, "lube_blend_min": 10}).json()
    assert ok["dissolution_pct"] - bad["dissolution_pct"] > 5 and bad["p_below_spec"] > ok["p_below_spec"]
    coarse = client.post("/v1/dissolution", headers=H, json={"api_d50_um": 90}).json()
    assert coarse["unfamiliar_conditions"]


def test_inspect_and_drift(client):
    from indl.data import images
    x, y = images.generate(40, 11, balanced=True)
    out = client.post("/v1/inspect", headers=H, json={"images": x.tolist()}).json()
    assert len(out) == 40 and sum((o["decision"] == "accept") == (t == 0) for o, t in zip(out, y, strict=True)) >= 36
    assert client.post("/v1/inspect", headers=H, json={"images": [[[0.5] * 10] * 10]}).status_code == 422
    xs, _ = images.generate(60, 12, shift=True)
    assert client.post("/v1/camera-drift", headers=H, json={"images": xs.tolist()}).json()["alarm"]


def test_rul_and_fleet(client):
    from indl.data import sensors
    g = sensors.generate(1, 99)
    rows = g.drop(columns=["unit", "rul", "life", "turret_rpm"]).fillna(0).to_dict("records")
    r = client.post("/v1/rul", headers=H, json={"rows": rows}).json()
    assert r["rul_cycles"] < 30 and r["alarm"]   # full history ends at failure
    fl = client.get("/v1/fleet", headers=H).json()
    assert len(fl) == 100 and fl[0]["rul_cycles"] <= fl[-1]["rul_cycles"]
