"""Yeast lifespans from the human per-frame classes of DetecDiv, the landmark rows and their label features.

DetecDiv classes: 1 unbudded, 2 small bud, 3 large bud, 4 dead, 5 empty trap, 6 clogged trap.
A lifespan starts at the first frame showing a cell and ends at the first later dead, empty or clogged frame;
it is an event when that frame is dead, otherwise censored there.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import FRAME_MIN, HORIZONS
from .constants import LANDMARK_STEP as STEP

LGB_SEED = 0                               # the LightGBM label rules; the survival fits use constants.SEED
UNBUD, SMALL, LARGE, DEAD, EMPTY, CLOG = 1, 2, 3, 4, 5, 6
LONG_CYCLE_FACTOR, LONG_CYCLE_BASE_CYCLES = 2.0, 5       # the "first long cycle" detector
#: LightGBM settings of the label rules: fixed, unweighted, one seed, deterministic, no tuning.
LGB_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 2,
              "deterministic": True, "force_col_wise": True}
NUM_ROUNDS = 200
DETECDIV_FEATURES = ("age_min", "divisions", "since_last_bud_min", "last_cycle_min", "first_long_cycle",
                     "since_first_long_cycle_min")


def window_targets(t: int, event_frame: int | None, last_obs: int, h_frames: int):
    """1 if the event falls in (t, t+h]; 0 if observed event-free through t+h; None if censored inside the horizon
    (left out, never a negative)."""
    if event_frame is not None and t < event_frame <= t + h_frames:
        return 1
    if event_frame is not None and event_frame > t + h_frames:
        return 0
    return 0 if last_obs >= t + h_frames else None


def detecdiv_lifespan(cls: np.ndarray):
    """(start, end frame, event) of one class series, or None when no frame shows a cell. The end is the first dead,
    empty or clogged frame after the start (the video length when there is none: censored)."""
    cell = np.isin(cls, (UNBUD, SMALL, LARGE))
    if not cell.any():
        return None
    start = int(np.argmax(cell))
    rest = np.where(np.isin(cls[start:], (DEAD, EMPTY, CLOG)))[0]
    if len(rest) == 0:
        return start, len(cls), False                       # alive to the end of the movie: censored
    end = start + int(rest[0])
    return start, end, bool(cls[end] == DEAD)


def detecdiv_buds(cls: np.ndarray, start: int, end: int) -> np.ndarray:
    """Frames of buds: a large-to-unbudded or large-to-small change inside the lifespan (the frame of the change)."""
    c = cls[start:end]
    idx = np.where((c[:-1] == LARGE) & np.isin(c[1:], (UNBUD, SMALL)))[0] + 1
    return idx + start


def detecdiv_features(t: int, start: int, buds: np.ndarray) -> dict:
    """Label features at landmark t from the buds at or before t. Age is the time since the lifespan's first
    frame; the first long cycle is a cycle at least twice the median of the first 5 cycles."""
    past = buds[buds <= t]
    f = {"age_min": (t - start) * FRAME_MIN, "divisions": len(past),
         "since_last_bud_min": float((t - past[-1]) * FRAME_MIN) if len(past) else float((t - start) * FRAME_MIN),
         "last_cycle_min": float((past[-1] - past[-2]) * FRAME_MIN) if len(past) >= 2 else np.nan,
         "first_long_cycle": 0, "since_first_long_cycle_min": np.nan}
    cyc = np.diff(past)
    if len(cyc) > LONG_CYCLE_BASE_CYCLES:                   # at least 6 closed cycles: 5 for the base, 1 to test
        base = float(np.median(cyc[:LONG_CYCLE_BASE_CYCLES]))
        later = np.where(cyc[LONG_CYCLE_BASE_CYCLES:] >= LONG_CYCLE_FACTOR * base)[0]
        if len(later):
            j = LONG_CYCLE_BASE_CYCLES + int(later[0]) + 1          # the bud that closes the first long cycle
            f["first_long_cycle"], f["since_first_long_cycle_min"] = 1, float((t - past[j]) * FRAME_MIN)
    return f


def detecdiv_rows(series: dict) -> tuple[pd.DataFrame, dict]:
    """One row per landmark of every lifespan ({trap: class array}), with the label features and the targets per
    horizon; and the lifespan counts."""
    rows = []
    notes = {"n_lifespans": 0, "n_traps_no_cell": 0, "n_deaths": 0, "n_lifespans_start_frame0": 0}
    for trap, cls in sorted(series.items()):
        ls = detecdiv_lifespan(cls)
        if ls is None:
            notes["n_traps_no_cell"] += 1
            continue
        start, end, event = ls
        notes["n_lifespans"] += 1
        notes["n_deaths"] += int(event)
        notes["n_lifespans_start_frame0"] += int(start == 0)
        buds = detecdiv_buds(cls, start, end)
        ev, last_obs = (end if event else None), end - 1
        pos = trap.rsplit("_", 1)[0]
        for t in range(start, end, STEP):
            r = {"unit": trap, "fold": pos, "t": t, "has_event": bool(event), **detecdiv_features(t, start, buds),
                 "frame_class": int(cls[t])}
            for h in HORIZONS:
                r[f"y_{h}"] = window_targets(t, ev, last_obs, h // FRAME_MIN)
            rows.append(r)
    return pd.DataFrame(rows), notes


def load_detecdiv(zpath: Path) -> dict:
    """{trap: per-frame human class array} from the `deep` series of each timeseries .mat file of the archive."""
    from scipy.io import loadmat
    out = {}
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if n.endswith(".mat") and "/timeseries/" in n:
                trap = n.rsplit("/", 1)[-1].removeprefix("lstm_labeled_").removesuffix(".mat")
                m = loadmat(io.BytesIO(z.read(n)), variable_names=["deep"])
                out[trap] = np.asarray(m["deep"]).ravel().astype(int)
    return out


def single_scores(trv: np.ndarray, ytr: np.ndarray, tev: np.ndarray, categorical: bool = False):
    """One feature as a rank score, its direction and missing-value side set on the training rows only.

    Numeric: the sign giving a training AUROC >= 0.5; a missing value goes below or above every training value,
    whichever gives the higher training AUROC (a tie goes below). Categorical: each class scores its training
    event rate. Returns (train scores, test scores, info), or (None, None, info) with no finite training value."""
    from .metrics import auroc
    if categorical:
        rate = pd.Series(ytr).groupby(trv).mean()
        prior = float(ytr.mean())
        return (pd.Series(trv).map(rate).fillna(prior).to_numpy(float),
                pd.Series(tev).map(rate).fillna(prior).to_numpy(float), {"kind": "categorical, training event rate"})
    fin = np.isfinite(trv)
    if not fin.any():
        return None, None, {"skipped": "no finite training value"}
    a = auroc(ytr[fin], trv[fin])
    sgn = -1.0 if (np.isfinite(a) and a < 0.5) else 1.0
    s_fin = sgn * trv[fin]
    best = None
    for side, val in (("below", float(s_fin.min()) - 1.0), ("above", float(s_fin.max()) + 1.0)):
        au = auroc(ytr, np.where(fin, sgn * np.nan_to_num(trv), val))
        if best is None or (np.isfinite(au) and (not np.isfinite(best[0]) or au > best[0])):
            best = (au, side, val)
    _, side, val = best
    s_tr = np.where(fin, sgn * np.nan_to_num(trv), val)
    te_fin = np.isfinite(tev)
    s_te = np.where(te_fin, sgn * np.nan_to_num(tev), val)
    return s_tr, s_te, {"direction": sgn, "missing_side": side}


def _ap_prepare(y: np.ndarray, s: np.ndarray):
    """Sort once by score (as auprc does) and mark the last row of each run of equal scores."""
    order = np.argsort(-s, kind="mergesort")
    ss = s[order]
    return order, y[order].astype(float), np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]


def _weighted_ap(prep, w: np.ndarray) -> float:
    """auprc on a resample in which row i appears w[i] times (one operating point per distinct score; a score run
    of weight 0 adds no recall)."""
    order, ys, ends = prep
    ws = w[order]
    tp = np.cumsum(ys * ws)[ends]
    called = np.cumsum(ws)[ends]
    if tp[-1] == 0:
        return float("nan")
    prec = np.divide(tp, called, out=np.zeros_like(tp), where=called > 0)
    return float((np.diff(np.r_[0.0, tp / tp[-1]]) * prec).sum())
