# Architecture

| Model | Input | Pipeline | Output |
|---|---|---|---|
| ANN | 11 process parameters + 3 in-process controls + site | median imputation + missing flags → standardise → 5 × MLP(64-64, SiLU) with mean/variance heads | dissolution, SD (noise + model), 90 % interval, P(< Q), sensitivity |
| CNN | 48 × 48 grayscale image | 3 conv blocks (BN, ReLU, max-pool) → global average pool → linear; temperature scaling | accept / reject, defect type, probabilities, Grad-CAM |
| RNN | 40 cycles × (6 sensors + missing rate) | per-recipe standardisation → forward fill → 2-layer LSTM(48) → MLP head | RUL (cycles), alarm, action |

Serving: `indl/service.py` loads the weights in `models/` once; FastAPI and Streamlit share it.
