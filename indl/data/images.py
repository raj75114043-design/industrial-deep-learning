"""Synthetic camera images from a tablet visual-inspection station (48 × 48 grayscale).

Classes: ``good``, ``chipped`` (edge material missing), ``cracked`` (thin fissure from the edge), ``spot`` (dark
foreign-particle contamination) and ``mottled`` (uneven colour from poor coating). Every image varies tablet position,
rotation of the score line, lighting direction, sensor noise and blur. Good tablets dominate (≈ 72 %), as on a real
line. A **line-2 camera** set has a dimmer, lower-contrast, blurrier sensor — a domain shift the model must survive.
"""
from __future__ import annotations

import numpy as np

CLASSES = ["good", "chipped", "cracked", "spot", "mottled"]
PRIORS = [0.72, 0.08, 0.08, 0.06, 0.06]
S = 48


def _blur(img: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return img
    ker = np.ones(k) / k
    img = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, img)
    return np.apply_along_axis(lambda c: np.convolve(c, ker, mode="same"), 0, img)


def render(r: np.random.Generator, cls: str, shift: bool = False) -> np.ndarray:
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    cx, cy = S / 2 + r.uniform(-3, 3), S / 2 + r.uniform(-3, 3)
    rad = r.uniform(14.5, 16.5)
    dist = np.hypot(xx - cx, yy - cy)
    tablet = dist <= rad
    ang = r.uniform(0, 2 * np.pi)
    light = 0.12 * (np.cos(ang) * (xx - cx) + np.sin(ang) * (yy - cy)) / rad
    img = np.full((S, S), 0.15) + r.normal(0, 0.01)
    img = np.where(tablet, 0.78 + light - 0.08 * (dist / rad) ** 4, img)
    th = r.uniform(0, np.pi)   # score line
    d_line = np.abs(np.cos(th) * (yy - cy) - np.sin(th) * (xx - cx))
    img = np.where(tablet & (d_line < 0.8) & (dist < rad - 3), img - 0.12, img)
    if cls == "chipped":
        a = r.uniform(0, 2 * np.pi)
        px, py = cx + rad * np.cos(a), cy + rad * np.sin(a)
        chip = np.hypot(xx - px, yy - py) < r.uniform(2.2, 4.5)
        img = np.where(chip & tablet, 0.15 + r.normal(0, 0.01), img)
    elif cls == "cracked":
        a = r.uniform(0, 2 * np.pi)
        crack_level = r.uniform(0.40, 0.58)
        length = r.uniform(5, 11)
        for t in np.linspace(0, length, 60):
            wob = 0.6 * np.sin(t * r.uniform(0.5, 1.2))
            x = cx + (rad - t) * np.cos(a) + wob * -np.sin(a)
            y = cy + (rad - t) * np.sin(a) + wob * np.cos(a)
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < S and 0 <= yi < S and tablet[yi, xi]:
                img[yi, xi] = crack_level
    elif cls == "spot":
        for _ in range(r.integers(1, 3)):
            a, rr = r.uniform(0, 2 * np.pi), r.uniform(0, rad - 4)
            sx, sy = cx + rr * np.cos(a), cy + rr * np.sin(a)
            img = np.where(np.hypot(xx - sx, yy - sy) < r.uniform(0.9, 1.9), r.uniform(0.3, 0.5), img)
    elif cls == "mottled":
        low = r.normal(0, 1, (6, 6))
        up = _blur(np.kron(low, np.ones((S // 6, S // 6))), 9)
        img = np.where(tablet, img - r.uniform(0.09, 0.13) * np.clip(up, -1.5, 1.5), img)
    img = _blur(img, 2 if r.random() < 0.3 else 1)
    img = img + r.normal(0, 0.03, (S, S))
    if shift:   # line-2 camera: dimmer, lower contrast, softer focus, noisier
        img = 0.35 + 0.75 * (img - 0.35) - 0.05
        img = _blur(img, 2) + r.normal(0, 0.035, (S, S))
    return np.clip(img, 0, 1).astype(np.float32)


def generate(n: int, seed: int, shift: bool = False, balanced: bool = False) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.default_rng(seed + (500 if shift else 0))
    p = None if balanced else PRIORS
    y = r.choice(len(CLASSES), n, p=p)
    x = np.stack([render(r, CLASSES[k], shift) for k in y])
    return x, y.astype(np.int64)
