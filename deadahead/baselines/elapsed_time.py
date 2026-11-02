"""The elapsed-time baseline: a weighted logistic regression on the time since the movie started.

Refit per held-out group on the other eight groups' rows with a determined label (baseline 2 of ``fit_predict``;
baseline 1 is the per-treatment event rate, baseline 3 the time since the track entered, identical here because
every track starts at frame 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare(w: pd.DataFrame) -> pd.DataFrame:
    """One evaluable row per (window, horizon), with the censored-in-horizon rows dropped. ``w`` holds one panel:
    the bundle's rows table (the primary panel without DU145_st_2), not the two-panel ``windows.parquet``."""
    sub = w[w["label"] != "CENSORED_IN_HORIZON"].copy()
    # time in movie is t_obs; time since track entry is t_obs - first_frame, and every track starts at frame 0
    sub["time_in_movie_min"] = sub["t_obs_frame"] * 3.0
    sub["time_since_entry_min"] = (sub["t_obs_frame"] - 0) * 3.0
    return sub


def fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, baseline: int, event_type: str
) -> np.ndarray:
    """Fit one baseline on ``train`` (label, window_weight and the baseline's column) and return the predicted
    probability of ``event_type`` for each row of ``test``."""
    y = (train["label"] == event_type).to_numpy(dtype=float)
    wgt = train["window_weight"].to_numpy(dtype=float)

    if baseline == 1:
        # per-treatment empirical rate, learned on the training groups only
        rate = {}
        for tr, g in train.groupby("treatment"):
            yy = (g["label"] == event_type).to_numpy(dtype=float)
            ww = g["window_weight"].to_numpy(dtype=float)
            rate[tr] = float(np.average(yy, weights=ww)) if ww.sum() > 0 else float(yy.mean())
        overall = float(np.average(y, weights=wgt)) if wgt.sum() > 0 else float(y.mean())
        return test["treatment"].map(rate).fillna(overall).to_numpy(dtype=float)

    col = "time_in_movie_min" if baseline == 2 else "time_since_entry_min"
    mu = float(train[col].mean())
    sd = float(train[col].std()) or 1.0
    Xtr = np.column_stack([np.ones(len(train)), (train[col].to_numpy() - mu) / sd])
    Xte = np.column_stack([np.ones(len(test)), (test[col].to_numpy() - mu) / sd])
    beta = _fit_logistic(Xtr, y, w=wgt)
    eta = np.clip(Xte @ beta, -30, 30)
    return 1.0 / (1.0 + np.exp(-eta))


def _fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
                  l2: float = 1e-6, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """Weighted logistic regression by Newton-IRLS; a small L2 term keeps the Hessian invertible under separation."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = X.T @ (w * (y - p)) - l2 * beta
        s = w * p * (1 - p)
        H = (X * s[:, None]).T @ X + l2 * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta
