"""The inputs of the DeadAhead head: landmark rows, training and evaluation labels, and the feature matrix.

DeadAhead reads, per landmark, the 1,024 features of the fine-tuned VideoMAE model followed by the 112 handcrafted
history features, plus the bin index (1,137 inputs per person-period row). Training labels are what is observed in
the movies truncated at frame 439; evaluation labels are read from the complete movies (``oracle.build_oracle``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deadahead.evaluation.rows import inner_groups
from deadahead.paths import tables_dir
from deadahead.survival.head import CAUSES
from deadahead.survival.oracle import build_oracle

KEY = ["movie_id", "track_id", "t_obs_frame"]

FEATURE_DIM = 1024                  # VideoMAE features per landmark
N_HANDCRAFTED = 112                 # 14 measures x 8 statistics
HORIZONS_MIN = (30, 60, 120, 240)
BIN_FRAMES, N_BINS = 10, 8          # 8 bins of 30 min
FRAME_MIN = 3.0
PANEL_NAME = "primary_439"
CUT_FRAME = 438                     # last frame of the truncated movies, 0-based
EXCLUDED_MOVIE = "DU145_st_2"       # left out of the primary panel, kept in the all_movies panel
PANELS = ("primary", "all_movies")
GROUPS = ("DU145_bp", "DU145_do", "DU145_st", "LNCaP_bp", "LNCaP_do", "LNCaP_st", "PNT1A_bp", "PNT1A_do", "PNT1A_st")
SEEDS = (2601562, 2601563, 2601564)

# Inputs per person-period row, the bin index included.
WIDTH = {"deadahead": FEATURE_DIM + N_HANDCRAFTED + 1, "videomae_adapted_survival": FEATURE_DIM + N_HANDCRAFTED + 1,
         "handcrafted_survival": N_HANDCRAFTED + 1}
METHODS = tuple(WIDTH)


# ------------------------------------------------------------------ tables
def load_tables(directory: Path | None = None, names=("windows", "events", "movies", "cells")) -> dict:
    """The windows, events, movies and cells tables (or those in ``names``) from ``tables_dir()`` or ``directory``."""
    d = Path(directory) if directory is not None else tables_dir()
    paths = {n: d / f"{n}.parquet" for n in names}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(p)
    return {n: pd.read_parquet(p) for n, p in paths.items()}


# ------------------------------------------------------------------ landmark rows and evaluation labels
def build_data(windows: pd.DataFrame, events: pd.DataFrame, movies: pd.DataFrame, hist: pd.DataFrame,
               cfg: dict, cfg_w: dict, panel: str) -> tuple[pd.DataFrame, list[str]]:
    """Landmark rows of the panel joined to their evaluation labels and to the 112 history features.

    One row per (landmark, horizon) whose outcome is known in the complete movie. The primary panel leaves out the
    movie in ``cfg["protocol"]["morphology_excluded_movies"]``; the all_movies panel keeps it. Returns the frame and the
    history columns in table order.
    """
    if panel not in PANELS:
        raise ValueError(panel)
    excl = list(cfg["protocol"]["morphology_excluded_movies"])
    w = windows[windows["panel"] == cfg["protocol"]["panel"]]
    if panel == "primary":
        w = w[~w["movie_id"].isin(excl)]
    oracle = build_oracle(w, events, movies, cfg_w)
    cols = [c for c in hist.columns if c not in ("track_id", "frame_index")]
    if len(cols) != N_HANDCRAFTED:
        raise AssertionError(f"{len(cols)} history features; {N_HANDCRAFTED} expected")
    data = oracle.merge(hist, left_on=["track_id", "t_obs_frame"], right_on=["track_id", "frame_index"], how="inner")
    return data, cols


def panel_windows(windows: pd.DataFrame, panel: str) -> pd.DataFrame:
    """The windows of the truncated movies, without the excluded movie on the primary panel."""
    w = windows[windows["panel"] == PANEL_NAME]
    return w[~w["movie_id"].isin([EXCLUDED_MOVIE])] if panel == "primary" else w


def split_fold(data: pd.DataFrame, fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training rows, held-out rows) for the held-out group ``fold``."""
    te = data[data["group"] == fold].copy()
    tr = data[data["group"] != fold].copy()
    if te.empty or tr.empty:
        raise AssertionError(f"fold {fold}: empty training or test frame")
    return tr, te


def inner_split(trx_outer: pd.DataFrame, datax: pd.DataFrame, g: str, h: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The frames of a cross-fitted inner head: the training rows of held-out group g outside group h (the seven
    groups other than g and h, in the row order of ``datax``), and the rows of h."""
    if h not in inner_groups(g):
        raise ValueError(f"{h} is not a training group of held-out group {g}")
    if (trx_outer["group"] == g).any():
        raise AssertionError(f"the training rows of held-out group {g} contain that group")
    tr = trx_outer[trx_outer["group"] != h]
    te = datax[datax["group"] == h]
    want = set(inner_groups(g)) - {h}
    got = set(tr["group"].unique())
    if got != want:
        raise AssertionError(f"inner fit ({g}, {h}): training groups {sorted(got)}, expected {sorted(want)}")
    if te.empty:
        raise AssertionError(f"inner fit ({g}, {h}): no rows of {h} to predict")
    return tr, te


def keys_of(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[KEY].reset_index(drop=True)


def assert_same_rows(before: pd.DataFrame, after: pd.DataFrame, what: str) -> None:
    """Raise unless ``after`` has the landmark keys of ``before``, row for row."""
    a, b = keys_of(before), keys_of(after)
    if len(a) != len(b) or not a.equals(b):
        raise AssertionError(f"{what}: the join changed the row order or the row count ({len(a)} -> {len(b)})")


# ------------------------------------------------------------------ training labels (what the truncated movies show)
def landmark_table(windows_panel: pd.DataFrame) -> pd.DataFrame:
    """One row per landmark of the panel with its time to the end of follow-up and the track's terminal event."""
    cols = KEY + ["time_to_event_min", "track_terminal_event"]
    lm = windows_panel[cols]
    per = lm.groupby(KEY, sort=False)[["time_to_event_min", "track_terminal_event"]].nunique()
    if (per > 1).any().any():
        raise AssertionError("a landmark's time_to_event_min or track_terminal_event differs across its horizons")
    return lm.drop_duplicates(subset=KEY).reset_index(drop=True)


def last_obs_frame(movie_ids: pd.Series, movies: pd.DataFrame, cut: int) -> np.ndarray:
    """The last observed frame of each movie in the panel: min(num_frames - 1, cut)."""
    n = dict(zip(movies["movie_id"], movies["num_frames"]))
    return np.minimum(movie_ids.map(n).to_numpy(dtype=float) - 1, float(cut))


def reference_inputs(lm: pd.DataFrame, untyped: str, movies: pd.DataFrame | None = None,
                     cut: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(time_to_event_min, track_terminal_event) per landmark.

    With ``untyped="censor_at_death"`` a death of unknown type is censored at the death; with ``"at_risk_to_cut"``
    the cell is followed up to the truncation frame and censored there.
    """
    tte = lm["time_to_event_min"].to_numpy(dtype=float).copy()
    term = lm["track_terminal_event"].to_numpy(dtype=object).copy()
    if untyped == "censor_at_death":
        return tte, term
    if untyped != "at_risk_to_cut":
        raise ValueError(untyped)
    if movies is None or cut is None:
        raise ValueError("at_risk_to_cut needs movies and the cut")
    u = term == "UNTYPED_DEATH"
    lo = last_obs_frame(lm["movie_id"], movies, cut)
    tte[u] = (lo[u] - lm["t_obs_frame"].to_numpy(dtype=float)[u]) * FRAME_MIN
    term[u] = "CENSORED"
    return tte, term


def panel_view(lm: pd.DataFrame, untyped: str = "censor_at_death", movies: pd.DataFrame | None = None,
               cut: int | None = None, bin_frames: int = BIN_FRAMES) -> pd.DataFrame:
    """The training targets of each landmark as observed in the truncated movies, in the head's input columns.

    D = time_to_event_min / 3 frames from the landmark to the end of its follow-up.
    * apoptosis or lytic death inside the panel: true_event = cause, t_event_true = t_obs + D, censor_offset = inf;
    * every other landmark: true_event = NONE, t_event_true = NaN, censor_offset = 10 * (ceil(D / 10) - 1) frames,
      so only the bins seen in full count as survived.
    """
    tte, term = reference_inputs(lm, untyped, movies, cut)
    D = tte / FRAME_MIN
    if not np.all(np.isfinite(D)) or not np.allclose(D, np.round(D), atol=0, rtol=0) or (D < 1).any():
        raise AssertionError("time_to_event_min is not a positive whole number of frames for every landmark")
    typed = np.isin(term, list(CAUSES))
    t_obs = lm["t_obs_frame"].to_numpy(dtype=float)
    out = lm[KEY].copy()
    out["true_event"] = np.where(typed, term, "NONE").astype(object)
    out["t_event_true"] = np.where(typed, t_obs + D, np.nan)
    out["censor_offset"] = np.where(typed, np.inf, bin_frames * (np.ceil(D / bin_frames) - 1))
    out["_untyped_inside"] = term == "UNTYPED_DEATH"
    return out


def apply_view(tr: pd.DataFrame, view: pd.DataFrame) -> pd.DataFrame:
    """Replace ``true_event`` and ``t_event_true`` and add ``censor_offset`` per landmark, keeping the row order."""
    base = tr.drop(columns=[c for c in ("true_event", "t_event_true", "censor_offset") if c in tr.columns])
    out = base.merge(view[KEY + ["true_event", "t_event_true", "censor_offset"]], on=KEY, how="left",
                     validate="many_to_one")
    assert_same_rows(tr, out, "the panel view")
    if out["censor_offset"].isna().any() or out["true_event"].isna().any():
        raise AssertionError(f"{int(out['censor_offset'].isna().sum())} training rows have no panel-view row")
    return out


def past_cut_count(view_train: pd.DataFrame, movies: pd.DataFrame, cut: int, bin_frames: int = BIN_FRAMES,
                   n_bins: int = N_BINS) -> int:
    """Training landmarks taught "no event" in a bin that ends after their movie's last observed frame: 0 for the
    panel view. ``view_train`` has one row per training landmark with the view's columns."""
    t_obs = view_train["t_obs_frame"].to_numpy(dtype=float)
    c = view_train["censor_offset"].to_numpy(dtype=float)
    dt = view_train["t_event_true"].to_numpy(dtype=float) - t_obs
    typed = view_train["true_event"].isin(list(CAUSES)).to_numpy() & np.isfinite(dt) & (dt > 0)
    ev_bin = np.where(typed, np.ceil(dt / bin_frames), np.inf)
    cens_bin = np.where(np.isfinite(c), np.ceil(c / bin_frames), np.inf)
    last = np.minimum(np.minimum(ev_bin, cens_bin), n_bins)
    survived = np.where(typed & (ev_bin <= n_bins), last - 1, last)        # bins taught "no event"
    lo = last_obs_frame(view_train["movie_id"], movies, cut)
    return int(((t_obs + bin_frames * survived) > lo).sum())


def assert_inside_cut(tr: pd.DataFrame, movies: pd.DataFrame, cut: int = CUT_FRAME) -> None:
    """Raise if a training landmark is at risk past its movie's last observed frame."""
    n = past_cut_count(tr.drop_duplicates(subset=KEY), movies, cut)
    if n != 0:
        raise AssertionError(f"{n} training landmarks are at risk past their movie's last observed frame")


# ------------------------------------------------------------------ VideoMAE features
def store_positions(frame: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """Each frame row's position in the feature row index ``rows``; raises on a key that is not in it."""
    if rows[KEY].duplicated().any():
        raise AssertionError("the feature row index has duplicated keys")
    idx = pd.MultiIndex.from_frame(rows[KEY].astype({"movie_id": str, "track_id": str, "t_obs_frame": np.int64}))
    q = pd.MultiIndex.from_frame(frame[KEY].astype({"movie_id": str, "track_id": str, "t_obs_frame": np.int64}))
    pos = idx.get_indexer(q)
    if (pos < 0).any():
        raise AssertionError(f"{int((pos < 0).sum())} frame rows are not in the feature row index")
    return pos


def feature_names(prefix: str, dim: int | None = None) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(FEATURE_DIM if dim is None else dim)]


def join_store(frame: pd.DataFrame, rows: pd.DataFrame, feats: np.ndarray, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    """Add the 1,024 features of each row (``feats`` row i belongs to ``rows`` row i), keeping the row order."""
    if feats.shape != (len(rows), FEATURE_DIM) or feats.dtype != np.float32:
        raise AssertionError(f"features {feats.shape} {feats.dtype}; ({len(rows)}, {FEATURE_DIM}) float32 expected")
    pos = store_positions(frame, rows)
    names = feature_names(prefix)
    block = pd.DataFrame(feats[pos], columns=names, index=frame.index)
    out = pd.concat([frame, block], axis=1)
    assert_same_rows(frame, out, f"the feature join ({prefix})")
    return out, names


def input_columns(method: str, video_cols: list[str], handcrafted_cols: list[str]) -> list[str]:
    """The head's input columns: the 1,024 VideoMAE features then the 112 handcrafted ones, or the 112 alone."""
    if method in ("deadahead", "videomae_adapted_survival"):
        cols = list(video_cols) + list(handcrafted_cols)
    elif method == "handcrafted_survival":
        cols = list(handcrafted_cols)
    else:
        raise ValueError(method)
    if len(cols) + 1 != WIDTH[method]:
        raise AssertionError(f"{method}: {len(cols) + 1} inputs per person-period row, {WIDTH[method]} expected")
    return cols


# ------------------------------------------------------------------ predictions
def coherence_violations(out: dict, horizons=HORIZONS_MIN) -> int:
    """Held-out landmarks whose incidence of either cause decreases from one horizon to the next (0 for this head)."""
    bad = np.zeros(len(out["_index"]), dtype=bool)
    for c in CAUSES:
        for h0, h1 in zip(horizons[:-1], horizons[1:]):
            bad |= np.asarray(out[h1][c]) < np.asarray(out[h0][c])
    return int(bad.sum())


def prediction_payload(out: dict, te: pd.DataFrame) -> dict:
    """The arrays of a prediction file.

    Per landmark: its keys and ``p_<cause>_<h>``, the incidence of each cause by each horizon. Per horizon:
    ``so_mask_<h>``, the landmarks whose outcome at h is known, and ``so_y_<h>``, their label (0 none, 1 apoptosis,
    2 lytic; -1 elsewhere).
    """
    idx = out["_index"]
    pay = {"index_movie": idx["movie_id"].to_numpy().astype(str), "index_track": idx["track_id"].to_numpy().astype(str),
           "index_tobs": idx["t_obs_frame"].to_numpy().astype(np.int64)}
    pos = pd.MultiIndex.from_frame(idx[KEY])
    n_bad = 0
    for h in HORIZONS_MIN:
        for c in CAUSES:
            p = np.asarray(out[h][c], dtype=np.float64)
            n_bad += int((~np.isfinite(p)).sum())
            pay[f"p_{c}_{h}"] = p
        teh = te[te["horizon_min"] == h]
        where = pos.get_indexer(pd.MultiIndex.from_frame(teh[KEY]))
        if (where < 0).any():
            raise AssertionError(f"h{h}: held-out rows without a prediction")
        mask = np.zeros(len(idx), dtype=bool)
        mask[where] = True
        y = np.full(len(idx), -1, dtype=np.int64)
        y[where] = teh["oracle_label"].to_numpy(dtype=np.int64)
        pay[f"so_mask_{h}"] = mask
        pay[f"so_y_{h}"] = y
    pay["n_nonfinite"] = np.int64(n_bad)
    return pay
