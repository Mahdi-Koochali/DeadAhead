"""The DeadAhead survival head: discrete-time competing risks fitted with LightGBM.

Each landmark is expanded into one row per 30-minute bin in which the cell is still at risk. A multiclass booster over
{no event, apoptosis, lytic death} gives the hazard of each cause in each bin, with the bin index as an input, and

    P(cause j by bin K) = sum_{k <= K} hazard_j(k) * S(k - 1),   S(k) = prod_{i <= k} (1 - sum_j hazard_j(i)),

which is non-decreasing in the horizon by construction, and the two causes sum to at most 1.
"""

from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

CAUSES = ["APOPTOSIS", "LYTIC"]
NO_EVENT = 0
CAUSE_IX = {"APOPTOSIS": 1, "LYTIC": 2}


def usable_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


# LightGBM threads: OMP_NUM_THREADS if set, else the CPUs this process may use. The released heads were fitted with
# 8 threads; use set_threads(8) to refit them.
_N_THREADS = int(os.environ.get("OMP_NUM_THREADS") or usable_cpus())

# Fixed settings, the same for every held-out group and seed (no hyperparameter search).
LGB_MULTI = {
    "objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
    "num_threads": _N_THREADS,
}
ROUNDS = 200


def set_threads(n: int) -> None:
    """Set the number of LightGBM threads the head uses."""
    LGB_MULTI["num_threads"] = int(n)


def train_predict_survival(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str],
                           cfg: dict, cfg_w: dict, seed: int,
                           interval_censored: bool = False,
                           ) -> dict[int, dict[str, np.ndarray]]:
    """Fit the head on the training landmarks ``tr`` and return the cumulative incidence of each cause on ``te``.

    ``tr`` needs ``true_event``, ``t_event_true``, ``censor_offset`` and ``window_weight``; ``cfg["survival_grid"]``
    gives the bin width and count. Returns {horizon_min: {cause: array}} and ``"_index"``, the keys of the unique
    test landmarks in prediction order; {} when no row is at risk or only one class is present.
    """
    interval = float(cfg_w["frame_interval_min"])
    bin_f = int(cfg["survival_grid"]["bin_frames"])
    n_bins = int(cfg["survival_grid"]["n_bins"])

    # one row per landmark (the head is not fitted per horizon)
    key = ["movie_id", "track_id", "t_obs_frame"]
    base = tr.drop_duplicates(subset=key).copy()

    # bin index of the event, relative to t_obs (inf when the cell has no event)
    dt = base["t_event_true"] - base["t_obs_frame"]
    ev_bin = np.where(base["true_event"].isin(CAUSES) & dt.notna() & (dt > 0),
                      np.ceil(dt / bin_f), np.inf)
    # optional interval-censored onset: a cell seen normal at t_onset_lower and dying by t_onset_upper spreads its
    # event over the bins [lo, hi]; bins before lo are survived and bins after hi are not at risk
    lo_bin = hi_bin = None
    if interval_censored and {"t_onset_lower", "t_onset_upper"} <= set(base.columns):
        dlo = base["t_onset_lower"] - base["t_obs_frame"]
        dhi = base["t_onset_upper"] - base["t_obs_frame"]
        lo_bin = np.where(dlo.notna() & (dlo > 0), np.ceil(dlo / bin_f), ev_bin)
        hi_bin = np.where(dhi.notna() & (dhi > 0), np.ceil(dhi / bin_f), ev_bin)
        lo_bin = np.minimum(lo_bin, ev_bin); hi_bin = np.maximum(hi_bin, ev_bin)
    # the bin in which observation stops (inf when never censored)
    c_off = base["censor_offset"].to_numpy()
    cens_bin = np.where(np.isfinite(c_off), np.ceil(c_off / bin_f), np.inf)

    X, Y, W = [], [], []
    Xb = base[cols].to_numpy(dtype=np.float32)
    wt = base["window_weight"].to_numpy(dtype=float)
    cause_ix = base["true_event"].map(CAUSE_IX).fillna(0).to_numpy(dtype=int)
    for k in range(1, n_bins + 1):
        # with an interval, a cell stays at risk until the top of its interval
        top = ev_bin if lo_bin is None else hi_bin
        at_risk = (top >= k) & (cens_bin >= k)             # still under observation in this bin
        if not at_risk.any():
            continue
        if lo_bin is None:
            lab = np.where(ev_bin[at_risk] == k, cause_ix[at_risk], NO_EVENT)
            ww = wt[at_risk]
        else:
            # inside the interval the event is labelled with its cause and weighted by 1 / (interval width), so a
            # cell's event weight sums to one; rows without an event never read their (nan) span
            with np.errstate(invalid="ignore"):
                span = np.maximum(hi_bin - lo_bin + 1.0, 1.0)
            inside = (lo_bin[at_risk] <= k) & (k <= hi_bin[at_risk]) & np.isfinite(ev_bin[at_risk])
            lab = np.where(inside, cause_ix[at_risk], NO_EVENT)
            ww = np.where(inside, wt[at_risk] / span[at_risk], wt[at_risk])
        X.append(np.column_stack([Xb[at_risk], np.full(at_risk.sum(), k, dtype=np.float32)]))
        Y.append(lab)
        W.append(ww)
    if not X:
        return {}
    Xp, Yp, Wp = np.vstack(X), np.concatenate(Y), np.concatenate(W)
    if len(np.unique(Yp)) < 2:
        return {}

    d = lgb.Dataset(Xp, label=Yp, weight=Wp)
    m = lgb.train({**LGB_MULTI, "seed": seed}, d, num_boost_round=ROUNDS)

    # hazards on the test landmarks, accumulated into the cumulative incidence
    te_base = te.drop_duplicates(subset=key)
    Xt = te_base[cols].to_numpy(dtype=np.float32)
    n = len(te_base)
    surv = np.ones(n)
    cum = {c: np.zeros(n) for c in CAUSES}
    out: dict[int, dict[str, np.ndarray]] = {}
    per_bin_h = bin_f * interval
    for k in range(1, n_bins + 1):
        p = m.predict(np.column_stack([Xt, np.full(n, k, dtype=np.float32)]))
        for c in CAUSES:
            cum[c] = cum[c] + surv * p[:, CAUSE_IX[c]]
        surv = surv * p[:, 0]
        h = int(round(k * per_bin_h))
        if h in cfg_w["horizons_min"]:
            out[h] = {c: cum[c].copy() for c in CAUSES}
    out["_index"] = te_base[key].reset_index(drop=True)
    return out
