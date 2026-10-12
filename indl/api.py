"""FastAPI service.

  POST /v1/dissolution        predicted dissolution, 90 % interval, P(below spec), parameter sensitivity
  POST /v1/inspect            tablet images (48 × 48 grayscale, values 0–1) → accept / reject + defect type
  POST /v1/camera-drift       batch image statistics vs the training reference
  POST /v1/rul                punch sensor history (≥ 1 row) → remaining useful life and alarm
  GET  /v1/fleet              RUL for every punch set in service
  GET  /v1/model              model card (evaluation results)
  GET  /health, /ready, /metrics
"""
from __future__ import annotations

import time
import uuid
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from indl import logging_setup
from indl import service as S
from indl.config import get_settings
from indl.data.images import S as SIZE
from indl.data.sensors import RECIPES, SENSORS

logging_setup.configure()
app = FastAPI(title="Industrial Deep Learning", version="1.0.0")
REQ = Counter("indl_http_requests_total", "HTTP requests", ["path", "status"])
LAT = Histogram("indl_request_seconds", "Request latency", ["path"])
DECISIONS = Counter("indl_inspection_decisions_total", "Inspection decisions", ["camera", "decision"])


@app.middleware("http")
async def ctx(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    tok = logging_setup.request_id.set(rid)
    t0 = time.perf_counter()
    try:
        resp = await call_next(request)
    finally:
        logging_setup.request_id.reset(tok)
    resp.headers["X-Request-ID"] = rid
    REQ.labels(request.url.path, resp.status_code).inc()
    LAT.labels(request.url.path).observe(time.perf_counter() - t0)
    return resp


def auth(x_api_key: str | None = Header(default=None)):
    if x_api_key not in get_settings().api_keys:
        raise HTTPException(401, "Invalid or missing API key")


class Batch(BaseModel):
    site: Literal["Goa", "Baddi", "Sikkim"] = "Goa"
    api_d50_um: float = Field(44.0, gt=0, lt=300)
    granulation_water_pct: float = Field(11.0, ge=0, le=30)
    granulation_time_min: float = Field(5.5, ge=0, le=30)
    impeller_rpm: float = Field(220.0, ge=0, le=1000)
    drying_lod_pct: float | None = Field(2.3, ge=0, le=10)
    mgst_pct: float = Field(1.0, ge=0, le=5)
    lube_blend_min: float = Field(5.0, ge=0, le=60)
    compression_force_kn: float = Field(14.0, ge=0, le=60)
    turret_rpm: float = Field(40.0, ge=0, le=150)
    coating_gain_pct: float = Field(3.0, ge=0, le=10)
    inlet_temp_c: float = Field(62.0, ge=0, le=120)
    hardness_kp: float | None = Field(9.7, ge=0, le=40)
    friability_pct: float | None = Field(0.3, ge=0, le=5)
    thickness_mm: float = Field(4.04, ge=0, le=20)


class Images(BaseModel):
    images: list[list[list[float]]] = Field(min_length=1, max_length=256)
    camera: Literal["line1", "line2"] = "line1"


class SensorRow(BaseModel):
    cycle: int
    recipe: str
    main_force_kn: float | None = None
    force_sd_kn: float | None = None
    ejection_force_n: float | None = None
    vibration_rms_g: float | None = None
    punch_temp_c: float | None = None
    weight_rsd_pct: float | None = None


class History(BaseModel):
    rows: list[SensorRow] = Field(min_length=1, max_length=2000)


def _arr(imgs: list) -> np.ndarray:
    a = np.asarray(imgs, dtype=np.float32)
    if a.ndim != 3 or a.shape[1:] != (SIZE, SIZE):
        raise HTTPException(422, f"images must be {SIZE}×{SIZE}")
    return np.clip(a, 0, 1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    b = S.bundle()
    return {"status": "ready", "models": ["dissolution_ensemble", "inspector_line1", "inspector_line2", "rul_lstm"], "fleet_units": int(b.fleet["unit"].nunique())}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/dissolution", dependencies=[Depends(auth)])
def dissolution(b: Batch):
    return S.predict_dissolution(b.model_dump())


@app.post("/v1/inspect", dependencies=[Depends(auth)])
def inspect(req: Images):
    out = S.inspect(_arr(req.images), req.camera)
    for o in out:
        DECISIONS.labels(req.camera, o["decision"]).inc()
    return out


@app.post("/v1/camera-drift", dependencies=[Depends(auth)])
def drift(req: Images):
    if len(req.images) < 30:
        raise HTTPException(422, "send at least 30 images for a drift check")
    return S.camera_drift(_arr(req.images))


@app.post("/v1/rul", dependencies=[Depends(auth)])
def rul(h: History):
    df = pd.DataFrame([r.model_dump() for r in h.rows])
    if not set(df["recipe"]) <= set(RECIPES):
        raise HTTPException(422, f"recipe must be one of {list(RECIPES)}")
    return S.predict_rul(df[["cycle", "recipe", *SENSORS]])


@app.get("/v1/fleet", dependencies=[Depends(auth)])
def fleet():
    return S.fleet_status().to_dict("records")


@app.get("/v1/model", dependencies=[Depends(auth)])
def model_card():
    return S.bundle().card
