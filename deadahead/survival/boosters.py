"""Cumulative incidence from a saved DeadAhead booster, and capture of the booster a fit trains."""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from deadahead.survival.frames import KEY
from deadahead.survival.head import CAUSE_IX, CAUSES, LGB_MULTI


def predict_incidence(booster, Xt: np.ndarray, cfg: dict, cfg_w: dict) -> dict:
    """Hazards per bin from ``booster``, accumulated into the cumulative incidence of each cause.

    ``Xt`` holds one row per landmark (float32, the head's input columns without the bin index). Returns
    {horizon_min: {cause: array}}, the same computation as the end of ``head.train_predict_survival``.
    """
    interval = float(cfg_w["frame_interval_min"])
    bin_f = int(cfg["survival_grid"]["bin_frames"])
    n_bins = int(cfg["survival_grid"]["n_bins"])
    n = len(Xt)
    surv = np.ones(n)
    cum = {c: np.zeros(n) for c in CAUSES}
    out: dict = {}
    per_bin_h = bin_f * interval
    for k in range(1, n_bins + 1):
        p = booster.predict(np.column_stack([Xt, np.full(n, k, dtype=np.float32)]),
                            num_threads=LGB_MULTI["num_threads"])
        for c in CAUSES:
            cum[c] = cum[c] + surv * p[:, CAUSE_IX[c]]
        surv = surv * p[:, 0]
        h = int(round(k * per_bin_h))
        if h in cfg_w["horizons_min"]:
            out[h] = {c: cum[c].copy() for c in CAUSES}
    return out


def base_matrix(te: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """The unique test landmarks and their input matrix, in the order ``train_predict_survival`` predicts them."""
    base = te.drop_duplicates(subset=KEY)
    return base[KEY].reset_index(drop=True), base[cols].to_numpy(dtype=np.float32)


def max_abs_diff(a: dict, b: dict) -> float:
    """Largest absolute difference over every (horizon, cause) of two incidence dicts; inf on a mismatch."""
    if set(a) - {"_index"} != set(b) - {"_index"}:
        return float("inf")
    m = 0.0
    for h in (k for k in a if k != "_index"):
        for c in a[h]:
            x, y = np.asarray(a[h][c], dtype=np.float64), np.asarray(b[h][c], dtype=np.float64)
            if x.shape != y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
                return float("inf")
            if x.size:
                m = max(m, float(np.abs(x - y).max()))
    return m


def capture_booster(call, *args, **kwargs):
    """Run ``call(*args, **kwargs)`` and collect every booster ``lightgbm.engine.train`` returns inside it.

    ``train_predict_survival`` does not return its booster; a profile hook reads it from the return of
    ``lightgbm.engine.train`` without changing either function. Returns (result, [boosters]).
    """
    import lightgbm.engine
    code = lightgbm.engine.train.__code__
    got: list = []

    def hook(frame, event, arg):
        if event == "return" and frame.f_code is code:
            got.append(arg)
    old = sys.getprofile()
    sys.setprofile(hook)
    try:
        res = call(*args, **kwargs)
    finally:
        sys.setprofile(old)
    return res, got


def load_booster(path: Path) -> lgb.Booster:
    """A booster saved as LightGBM model text."""
    return lgb.Booster(model_file=str(path))


def save_booster(booster: lgb.Booster, path: Path) -> Path:
    """Save ``booster`` as LightGBM model text (written whole, then renamed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    booster.save_model(str(tmp))
    tmp.replace(path)
    return path
