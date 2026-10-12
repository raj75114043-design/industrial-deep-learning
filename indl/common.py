"""Shared training utilities: deterministic seeding, a generic mini-batch loop with early stopping, and calibration."""
from __future__ import annotations

import copy
import random
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit(model: nn.Module, train: tuple, val: tuple, loss_fn: Callable, epochs: int = 30, lr: float = 2e-3, batch: int = 128,
        weight_decay: float = 1e-4, patience: int = 5, augment: Callable | None = None, seed: int = 0) -> dict:
    """Adam + cosine schedule, keeps the weights with the best validation loss."""
    seed_everything(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xs, ys = train
    n = len(ys)
    best, best_state, bad, hist = float("inf"), None, 0, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = xs[idx], ys[idx]
            if augment is not None:
                xb = augment(xb, g)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(val[0]), val[1]))
        hist.append({"epoch": ep + 1, "train_loss": tot / n, "val_loss": vl})
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return {"best_val_loss": best, "epochs_run": len(hist), "history": hist}


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Temperature scaling (Guo et al. 2017): one scalar that minimises validation NLL."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().exp())


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of the top-class confidence."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(e)
