"""CNN for tablet visual inspection.

* Architecture — 3 convolutional blocks (conv-BN-ReLU ×2, max-pool), global average pooling, linear head (~41 k weights).
* Training — class-weighted cross-entropy (defects are rare), geometric augmentation (flips, 90° rotations) and
  photometric augmentation (brightness, contrast, blur, noise) so a different camera does not break the model.
* Calibration — temperature scaling on the validation set.
* Decision — a tablet is *accepted* only if P(good) ≥ τ, with τ set on validation so that ≤ 0.5 % of defective tablets
  are accepted (a margin for a ≤ 1.5 % escape gate on unseen data) (escapes cost far more than re-inspecting a good tablet). Everything else is rejected with its most
  likely defect type.
* Camera drift — batch brightness / contrast / sharpness are compared with the training reference; a new camera is
  handled by fine-tuning on a few hundred labelled images and re-calibrating.
* Explanation — Grad-CAM heat-map of the last convolutional block.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from indl.common import fit, fit_temperature, seed_everything
from indl.data.images import CLASSES


class TabletCNN(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), width: int = 12):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.b1, self.b2, self.b3 = block(1, width), block(width, 2 * width), block(2 * width, 4 * width)
        self.head = nn.Linear(4 * width, n_classes)
        self.drop = nn.Dropout(0.1)

    def features(self, x):
        x = F.max_pool2d(self.b1(x), 2)
        x = F.max_pool2d(self.b2(x), 2)
        return self.b3(x)

    def forward(self, x):
        f = self.features(x)
        return self.head(self.drop(f.mean((2, 3))))


def augment(photometric: bool):
    def fn(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        n = x.shape[0]
        if torch.rand(1, generator=g) < 0.5:
            x = x.flip(3)
        if torch.rand(1, generator=g) < 0.5:
            x = x.flip(2)
        k = int(torch.randint(0, 4, (1,), generator=g))
        x = torch.rot90(x, k, (2, 3))
        if photometric:
            b = (torch.rand(n, 1, 1, 1, generator=g) - 0.5) * 0.2
            c = 0.7 + torch.rand(n, 1, 1, 1, generator=g) * 0.45
            x = (x - 0.4) * c + 0.4 + b
            if torch.rand(1, generator=g) < 0.3:
                x = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
            x = x + torch.randn(x.shape, generator=g) * (torch.rand(n, 1, 1, 1, generator=g) * 0.04)
        return x
    return fn


@dataclass
class Inspector:
    model: TabletCNN
    temperature: float = 1.0
    tau_good: float = 0.5
    meta: dict = field(default_factory=dict)

    def probs(self, x: np.ndarray, calibrated: bool = True) -> np.ndarray:
        t = torch.tensor(x[:, None] if x.ndim == 3 else x, dtype=torch.float32)
        with torch.no_grad():
            logits = torch.cat([self.model(t[i:i + 512]) for i in range(0, len(t), 512)])
        return torch.softmax(logits / (self.temperature if calibrated else 1.0), 1).numpy()

    def decide(self, p: np.ndarray) -> np.ndarray:
        """0 = accept as good; otherwise the predicted defect class index (never 0)."""
        defect = p[:, 1:].argmax(1) + 1
        return np.where(p[:, 0] >= self.tau_good, 0, defect)

    def gradcam(self, img: np.ndarray, cls: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        x = torch.tensor(img[None, None], dtype=torch.float32)
        feats = self.model.features(x)
        feats.retain_grad()
        logits = self.model.head(feats.mean((2, 3)))
        c = int(logits.argmax()) if cls is None else cls
        logits[0, c].backward()
        w = feats.grad.mean((2, 3), keepdim=True)
        cam = F.relu((w * feats).sum(1))[0]
        cam = F.interpolate(cam[None, None], size=img.shape, mode="bilinear", align_corners=False)[0, 0].detach().numpy()
        self.model.zero_grad()
        return cam / (cam.max() + 1e-8), torch.softmax(logits / self.temperature, 1).detach().numpy()[0]

    def state(self) -> dict:
        return {"model": self.model.state_dict(), "temperature": self.temperature, "tau_good": self.tau_good, "meta": self.meta}

    @classmethod
    def from_state(cls, s: dict) -> Inspector:
        m = TabletCNN()
        m.load_state_dict(s["model"])
        m.eval()
        return cls(m, s["temperature"], s["tau_good"], s.get("meta", {}))


def choose_tau(p_val: np.ndarray, y_val: np.ndarray, max_escape: float = 0.005) -> float:
    """Smallest acceptance threshold on P(good) whose escape rate on validation is ≤ max_escape."""
    defect = y_val != 0
    for tau in np.linspace(0.5, 0.999, 500):
        if ((p_val[:, 0] >= tau) & defect).sum() / max(defect.sum(), 1) <= max_escape:
            return float(tau)
    return 0.999


def train_inspector(xtr, ytr, xva, yva, photometric: bool = True, epochs: int = 16, seed: int = 0) -> Inspector:
    seed_everything(seed)
    model = TabletCNN()
    w = torch.tensor(len(ytr) / (len(CLASSES) * np.bincount(ytr, minlength=len(CLASSES))), dtype=torch.float32).sqrt()
    loss = nn.CrossEntropyLoss(weight=w)
    t = lambda a: torch.tensor(a[:, None], dtype=torch.float32)  # noqa: E731
    h = fit(model, (t(xtr), torch.tensor(ytr)), (t(xva), torch.tensor(yva)), loss, epochs=epochs, lr=2e-3, batch=64, patience=10,
            augment=augment(photometric), seed=seed)
    with torch.no_grad():
        logits = model(t(xva))
    temp = fit_temperature(logits, torch.tensor(yva))
    insp = Inspector(model, temp, meta={"epochs_run": h["epochs_run"], "photometric_augmentation": photometric, "params": sum(p.numel() for p in model.parameters())})
    insp.tau_good = choose_tau(insp.probs(xva), yva)
    return insp


def image_stats(x: np.ndarray) -> np.ndarray:
    """Per-image brightness, contrast and sharpness (mean |Laplacian|) — inputs to the camera-drift check."""
    lap = np.abs(4 * x[:, 1:-1, 1:-1] - x[:, :-2, 1:-1] - x[:, 2:, 1:-1] - x[:, 1:-1, :-2] - x[:, 1:-1, 2:]).mean((1, 2))
    return np.stack([x.mean((1, 2)), x.std((1, 2)), lap], 1)


def drift_check(ref: np.ndarray, x: np.ndarray, z_alarm: float = 6.0) -> dict:
    """Compare a batch of images with the training reference; alarm when any statistic moves by > z_alarm standard errors."""
    s = image_stats(x)
    mu, sd = ref.mean(0), ref.std(0) + 1e-9
    z = (s.mean(0) - mu) / (sd / np.sqrt(len(s)))
    names = ["brightness", "contrast", "sharpness"]
    return {"z": {n: round(float(v), 1) for n, v in zip(names, z, strict=True)}, "alarm": bool(np.abs(z).max() > z_alarm),
            "signals": [n for n, v in zip(names, z, strict=True) if abs(v) > z_alarm]}


def adapt(base: Inspector, x_new, y_new, x_replay, y_replay, x_val, y_val, epochs: int = 8, seed: int = 0) -> Inspector:
    """Fine-tune on a small labelled set from the new camera (plus replay of old images to avoid forgetting), then
    re-calibrate the temperature and the acceptance threshold on new-camera validation images."""
    import copy
    seed_everything(seed)
    model = copy.deepcopy(base.model)
    x = np.concatenate([x_new, x_replay])
    y = np.concatenate([y_new, y_replay])
    t = lambda a: torch.tensor(a[:, None], dtype=torch.float32)  # noqa: E731
    fit(model, (t(x), torch.tensor(y)), (t(x_val), torch.tensor(y_val)), nn.CrossEntropyLoss(), epochs=epochs, lr=5e-4, batch=64, patience=4,
        augment=augment(True), seed=seed)
    with torch.no_grad():
        logits = model(t(x_val))
    ins = Inspector(model, fit_temperature(logits, torch.tensor(y_val)), meta={**base.meta, "adapted_with": len(y_new)})
    ins.tau_good = choose_tau(ins.probs(x_val), y_val)
    return ins
