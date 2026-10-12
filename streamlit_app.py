"""Tablet manufacturing deep-learning console (Streamlit)."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from indl import sequence as Q  # noqa: E402
from indl import service as S  # noqa: E402
from indl.data import images  # noqa: E402
from indl.data.sensors import SENSORS  # noqa: E402

logging.disable(logging.INFO)
st.set_page_config(page_title="Industrial Deep Learning", page_icon="💊", layout="wide")
EV = json.loads(Path("reports/eval.json").read_text()) if Path("reports/eval.json").exists() else {}
B = S.bundle()

with st.sidebar:
    st.markdown("### 💊 Industrial Deep Learning")
    st.caption("Meridian Pharma (fictional) · immediate-release tablet line")
    st.markdown("**ANN** — dissolution from process parameters  \n**CNN** — visual inspection of tablets  \n**RNN** — remaining life of press punches")
    if EV:
        st.markdown(f"Dissolution RMSE **{EV['ann_rmse']:.2f} %** · inspection accuracy **{EV['cnn_accuracy']:.1%}** · RUL RMSE **{EV['rnn_rmse_last']:.1f}** cycles")
    st.caption("All batches, images and sensor logs are synthetic. Models are PyTorch, CPU-only.")

tabs = st.tabs(["💊 Dissolution (ANN)", "🔍 Visual inspection (CNN)", "⚙️ Punch life (RNN)", "🧪 Evaluation", "🏗️ Architecture"])

with tabs[0]:
    st.markdown("Predict % dissolved at 30 min (spec **Q = 80 %**) before the lab result is available, with uncertainty from a 5-network deep ensemble.")
    c = st.columns(4)
    b = dict(S.DEFAULT_BATCH)
    b["site"] = c[0].selectbox("Site", ["Goa", "Baddi", "Sikkim"])
    b["api_d50_um"] = c[1].slider("API particle size D50 (µm)", 25.0, 100.0, 44.0, 1.0)
    b["granulation_water_pct"] = c[2].slider("Granulation water (%)", 8.0, 14.0, 11.0, 0.1)
    b["drying_lod_pct"] = c[3].slider("Drying LOD (%)", 0.8, 4.0, 2.3, 0.1)
    c = st.columns(4)
    b["mgst_pct"] = c[0].slider("Mg stearate (%)", 0.5, 1.5, 1.0, 0.05)
    b["lube_blend_min"] = c[1].slider("Lubrication blend time (min)", 2.0, 10.0, 5.0, 0.5)
    b["compression_force_kn"] = c[2].slider("Compression force (kN)", 8.0, 21.0, 14.0, 0.5)
    b["coating_gain_pct"] = c[3].slider("Coating weight gain (%)", 2.0, 4.0, 3.0, 0.1)
    b["hardness_kp"] = 2 + 0.55 * b["compression_force_kn"] - 0.6 * (b["drying_lod_pct"] - 2.3)
    r = S.predict_dissolution(b)
    k = st.columns(4)
    k[0].metric("Predicted dissolution", f"{r['dissolution_pct']:.1f} %")
    k[1].metric("90 % interval", f"{r['interval90'][0]:.0f} – {r['interval90'][1]:.0f} %")
    k[2].metric("P(below Q = 80 %)", f"{r['p_below_spec']:.1%}")
    k[3].metric("Hardness (estimated IPC)", f"{b['hardness_kp']:.1f} kP")
    if r["unfamiliar_conditions"]:
        st.warning(f"Unfamiliar process conditions — the ensemble members disagree (model SD {r['model_uncertainty_sd']:.1f} %). Confirm with a lab test.")
    left, right = st.columns(2)
    sens = pd.Series(r["sensitivity_per_5pct_increase"]).rename("Δ dissolution (%)").reset_index().rename(columns={"index": "parameter"})
    left.altair_chart(alt.Chart(sens).mark_bar().encode(x="Δ dissolution (%):Q", y=alt.Y("parameter:N", sort="x")).properties(
        height=320, title="Effect of a +5 % change in each parameter"), width="stretch")
    gx, gy = np.linspace(0.5, 1.5, 21), np.linspace(2, 10, 21)
    grid = pd.DataFrame([{**b, "mgst_pct": x, "lube_blend_min": y} for y in gy for x in gx])
    pr = B.ensemble.predict(grid)
    grid["dissolution"] = pr["pred"].round(1)
    right.altair_chart(alt.Chart(grid).mark_rect().encode(
        x=alt.X("mgst_pct:O", title="Mg stearate %", axis=alt.Axis(values=[0.5, 0.75, 1.0, 1.25, 1.5], format=".2f")),
        y=alt.Y("lube_blend_min:O", title="Blend time (min)", sort="descending", axis=alt.Axis(values=[2.0, 4.0, 6.0, 8.0, 10.0])),
        color=alt.Color("dissolution:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[70, 100])), tooltip=["mgst_pct", "lube_blend_min", "dissolution"],
    ).properties(height=320, title="Design space: over-lubrication (other parameters as set above)"), width="stretch")

with tabs[1]:
    c1, c2, c3 = st.columns(3)
    camera = c1.selectbox("Camera", ["line1", "line2"], format_func=lambda x: {"line1": "Line 1 (training camera)", "line2": "Line 2 (new camera)"}[x])
    model_for = c2.selectbox("Model", ["line1", "line2"], format_func=lambda x: {"line1": "base model", "line2": "adapted to line 2"}[x], index=0)
    seed = c3.number_input("Sample seed", 0, 9999, 7)
    x, y = images.generate(16, int(seed), shift=camera == "line2", balanced=True)
    res = S.inspect(x, model_for)
    cols = st.columns(8)
    for i in range(16):
        ok = (res[i]["decision"] == "accept") == (y[i] == 0)
        lab = "✅ accept" if res[i]["decision"] == "accept" else f"⛔ {res[i]['defect']}"
        cols[i % 8].image(x[i], caption=f"{lab} {'' if ok else '❗'}\ntrue: {images.CLASSES[y[i]]}", width="stretch", clamp=True)
    pick = st.slider("Explain image #", 1, 16, 1) - 1
    cam, prob = B.inspectors[model_for].gradcam(x[pick])
    rgb = np.stack([x[pick]] * 3, -1) * 0.6
    rgb[..., 0] = np.clip(rgb[..., 0] + 0.8 * cam, 0, 1)
    a, b2 = st.columns([1, 2])
    a.image(rgb, caption="Grad-CAM: where the network looked", width=220, clamp=True)
    b2.altair_chart(alt.Chart(pd.DataFrame({"class": images.CLASSES, "p": prob})).mark_bar().encode(x="p:Q", y=alt.Y("class:N", sort=None)).properties(height=200),
                    width="stretch")
    xb, _ = images.generate(200, int(seed) + 1, shift=camera == "line2")
    dr = S.camera_drift(xb)
    (st.error if dr["alarm"] else st.success)(f"Camera drift check on 200 images: {'ALARM — ' + ', '.join(dr['signals']) if dr['alarm'] else 'no drift'} · z = {dr['z']}")

with tabs[2]:
    fl = S.fleet_status()
    a1, a2, a3 = st.columns(3)
    a1.metric("Punch sets in service", len(fl))
    a2.metric("Alarms (plan change)", int(fl["alarm"].sum()))
    a3.metric("Alarm threshold", f"RUL ≤ {B.rul_threshold:.0f} cycles")
    st.dataframe(fl.head(15), hide_index=True, width="stretch")
    unit = st.selectbox("Unit", fl["unit"].tolist())
    g = B.fleet[B.fleet["unit"] == unit]
    xw, yw, _ = Q.windows(g, B.rul.scaler)
    traj = pd.DataFrame({"cycle": g["cycle"].iloc[Q.WINDOW - 1:].to_numpy()[:len(yw)] if len(g) >= Q.WINDOW else g["cycle"].iloc[-len(yw):].to_numpy(),
                         "predicted": B.rul.predict(xw), "true": yw}).melt("cycle", var_name="series", value_name="RUL")
    st.altair_chart(alt.Chart(traj).mark_line().encode(x="cycle:Q", y="RUL:Q", color="series:N").properties(height=260, title="Remaining useful life"),
                    width="stretch")
    sens = g.melt(["cycle", "recipe"], SENSORS, var_name="sensor", value_name="value")
    st.altair_chart(alt.Chart(sens).mark_line(strokeWidth=1).encode(x="cycle:Q", y=alt.Y("value:Q", scale=alt.Scale(zero=False)), color="recipe:N").properties(
        height=110, width=330).facet("sensor:N", columns=3).resolve_scale(y="independent"))
    st.caption("Sensor levels jump with the product recipe; the network sees recipe-normalised signals, so only wear remains.")

with tabs[3]:
    if not EV:
        st.info("Run `python -m indl.eval.run --save` first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.markdown("**ANN — dissolution (RMSE, % points)**")
        c1.dataframe(pd.DataFrame({"in-distribution": [EV["ann_rmse"], EV["gbm_rmse"], EV["ridge_rmse"], EV["ann_noise_floor_rmse"]],
                                   "new API supplier": [EV["ann_ood_rmse"], EV["gbm_ood_rmse"], None, None]},
                                  index=["deep ensemble", "gradient boosting", "ridge", "noise floor"]).round(2), width="stretch")
        c1.caption(f"90 % interval coverage {EV['ann_coverage90']:.0%} (new supplier: {EV['ann_ood_coverage90']:.0%}); "
                   f"model uncertainty ×{EV['ann_ood_uncertainty_ratio']:.1f} on the new supplier (AUROC {EV['ann_ood_auroc']:.2f}).")
        c2.markdown("**CNN — inspection**")
        c2.dataframe(pd.DataFrame({"value": [EV["cnn_accuracy"], EV["cnn_macro_f1"], EV["cnn_escape_rate"], EV["cnn_false_reject"], EV["cnn_ece_uncalibrated"],
                                             EV["cnn_ece"], EV["cnn_shift_accuracy"]["without"], EV["cnn_shift_accuracy"]["with_photometric_aug"],
                                             EV["cnn_adapted_accuracy"], EV["cnn_adapted_false_reject"]]},
                                  index=["accuracy", "macro-F1", "escape rate", "false reject", "ECE raw", "ECE calibrated", "line 2, no photometric aug",
                                         "line 2, with aug", "line 2, adapted", "line 2 adapted, false reject"]).round(4), width="stretch")
        c3.markdown("**RNN — punch RUL (last window per unit)**")
        c3.dataframe(pd.DataFrame(EV["rnn_results"]).T[["rmse_last", "nasa"]].round(1), width="stretch")
        c3.caption(f"Early warning (RUL ≤ 25): recall {EV['rnn_warning_recall']:.0%}, precision {EV['rnn_warning_precision']:.0%}")

with tabs[4]:
    st.graphviz_chart("""
    digraph A { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f4f4f4", fontname=Helvetica];
      subgraph cluster_a { label="ANN"; P [label="Batch record\\nprocess params + IPC"]; O [label="Outlier screen\\n(lab errors)"];
        E [label="5 × Gaussian MLP\\ndeep ensemble"];
        D [label="Dissolution ± SD\\nP(OOS) · design space"]; P -> O -> E -> D; }
      subgraph cluster_c { label="CNN"; I [label="Camera image\\n48×48"]; C [label="CNN + augmentation\\ntemperature scaling"];
        R [label="Accept / reject\\nτ for ≤0.5% escapes"];
        G [label="Grad-CAM"]; W [label="Drift check →\\nfew-shot adaptation"]; I -> C -> R; C -> G; I -> W -> C; }
      subgraph cluster_r { label="RNN"; S [label="Press sensors\\nper cycle"]; N [label="Recipe\\nnormalisation"]; L [label="LSTM\\nasymmetric loss"];
        U [label="RUL + alarm\\nplan tool change"]; S -> N -> L -> U; } }""")
