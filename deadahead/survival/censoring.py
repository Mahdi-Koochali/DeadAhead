"""The injected-censoring experiment: does handling censored cells correctly matter when censoring is added?

40 % administrative censoring (a censor time drawn independently of every covariate) is injected into the training
landmarks of each held-out group. Four regimes are trained on the same censored data with the 112 handcrafted
history features and scored on the held-out group against the oracle labels of the complete movies:

    (a) a_naive_drop       drop every landmark that is censored at any time
    (b) b_naive_mislabel   label a window censored inside its horizon as event-free
    (c) c_censoring_aware  drop only the windows censored inside their horizon
    (d) d_joint_survival   the DeadAhead survival head (deadahead/survival/head.py). The bin holding an injected censor
                           is kept, so a death later in that bin can stay a target and the bin can count as survived.
                           This arm is not reported: no number in the paper uses it.

The cells whose death has no type are left out of training and testing. One unit is one (held-out group, regime,
seed): 9 x 4 x 3 = 108 units. ``deadahead reproduce --part censoring`` scores the stored units.

    deadahead censoring-experiment (--group G | --all) [--regime R] [--seed S] [--tables-dir DIR] [--history FILE]
                                   [--threads N] [--out-dir DIR]

Writes <out-dir>/units/<group>__<regime>__s<seed>.npz (y_<h>, p_<regime>_<cause>_<h>, and for (d) its landmark
index and masks) and a .json beside it (achieved censoring rate, the risk-set counts); a unit already written is
skipped, so a rerun resumes. <out-dir>/cohort.json holds the cohort counts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from deadahead.evaluation.rows import GROUPS
from deadahead.paths import runs_dir
from deadahead.survival import head
from deadahead.survival.frames import KEY, load_tables
from deadahead.survival.head import CAUSE_IX, CAUSES, NO_EVENT, train_predict_survival
from deadahead.survival.oracle import build_oracle

UNTYPED = "UNTYPED_DEATH"
REGIMES = ("a_naive_drop", "b_naive_mislabel", "c_censoring_aware", "d_joint_survival")
BINARY = REGIMES[:3]
#: the injected censoring: mechanism and the fraction of at-risk (landmark, horizon) training rows censored
LEVEL = ("i_administrative", 0.40)
#: the binary classifiers of regimes (a) to (c); the survival regime uses the head's own settings
LGB_BINARY = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
    "num_threads": 1,
}
ROUNDS = 200
#: the released units were trained with one LightGBM thread
DEFAULT_THREADS = 1


def _mechanism_score(landmarks: pd.DataFrame, mechanism: str, seed: int) -> np.ndarray:
    """A score that orders the landmarks for censoring (only the order is used): uniform random under the
    administrative mechanism, independent of every covariate."""
    n = len(landmarks)
    rng = np.random.default_rng(seed)
    if mechanism == "i_administrative":
        return rng.random(n)

    raise ValueError(f"unknown mechanism {mechanism}")


def inject_censor_time(landmarks: pd.DataFrame, horizons: pd.DataFrame, rate: float,
                       mechanism: str, seed: int, max_horizon_frames: int,
                       interval: float, tol: float = 0.004) -> tuple[np.ndarray, float]:
    """Draw a censor time per landmark, calibrated by bisection to censor ``rate`` of the at-risk (landmark,
    horizon) rows: the fraction whose observation stops inside the horizon with no event seen by then.

    Returns (censor offsets in frames after t_obs, inf = never censored; the achieved rate)."""
    n = len(landmarks)
    if rate <= 0 or n == 0:
        return np.full(n, np.inf), 0.0

    score = _mechanism_score(landmarks, mechanism, seed)
    order = np.argsort(-score, kind="mergesort")
    draws = np.random.default_rng(seed + 1).uniform(1.0, max_horizon_frames, size=n)

    key = ["movie_id", "track_id", "t_obs_frame"]
    pos = pd.Series(np.arange(n), index=pd.MultiIndex.from_frame(landmarks[key]))
    row_ix = pos.reindex(pd.MultiIndex.from_frame(horizons[key])).to_numpy()
    hf = (horizons["horizon_min"] / interval).to_numpy()
    ev_off = (horizons["t_event_true"] - horizons["t_obs_frame"]).to_numpy()

    def achieved_for(q: float) -> tuple[np.ndarray, float]:
        k = int(round(q * n))
        co = np.full(n, np.inf)
        if k > 0:
            co[order[:k]] = draws[order[:k]]
        c = co[row_ix]
        censored_in_h = (c < hf) & ~(np.isfinite(ev_off) & (ev_off <= c))
        return co, float(censored_in_h.mean())

    lo, hi = 0.0, 1.0
    best, best_rate = achieved_for(1.0)
    for _ in range(40):
        mid = (lo + hi) / 2
        co, got = achieved_for(mid)
        if abs(got - rate) < abs(best_rate - rate):
            best, best_rate = co, got
        if abs(got - rate) <= tol:
            return co, got
        if got < rate:
            lo = mid
        else:
            hi = mid
    return best, best_rate


def regime_training_view(tr: pd.DataFrame, regime: str, horizon_frames: int) -> pd.DataFrame:
    """The training rows one regime sees at one horizon, with its labels: (a) drops a landmark censored at any
    time, (b) labels a window censored inside its horizon event-free, (c) drops only the windows censored inside
    their horizon."""
    c_off = tr["censor_offset"].to_numpy()
    ev_off = (tr["t_event_true"] - tr["t_obs_frame"]).to_numpy()
    event_first = np.isfinite(ev_off) & (ev_off <= c_off)
    censored_in_horizon = (c_off < horizon_frames) & ~event_first
    censored_ever = np.isfinite(c_off) & ~event_first

    d = tr.copy()
    if regime == "a_naive_drop":
        return d[~censored_ever]
    if regime == "b_naive_mislabel":
        d.loc[censored_in_horizon, "oracle_label"] = NO_EVENT
        return d
    if regime == "c_censoring_aware":
        return d[~censored_in_horizon]
    raise ValueError(regime)


def train_predict_horizon(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str], cause: str,
                          regime: str, horizon_frames: int, seed: int) -> np.ndarray:
    """One binary LightGBM classifier per regime, horizon and cause on the regime's training view; returns the
    held-out scores."""
    view = regime_training_view(tr, regime, horizon_frames)
    if view.empty:
        return np.zeros(len(te))
    y = (view["oracle_label"] == CAUSE_IX[cause]).to_numpy(dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        return np.full(len(te), float(y.mean()))
    d = lgb.Dataset(view[cols], label=y, weight=view["window_weight"])
    m = lgb.train({**LGB_BINARY, "seed": seed}, d, num_boost_round=ROUNDS)
    return m.predict(te[cols])


def untyped_death_tracks(events: pd.DataFrame) -> set:
    """The cells whose death has no type."""
    return set(events.loc[events["event_type"] == UNTYPED, "track_id"].astype(str))


def declared_cohort(data: pd.DataFrame, tracks: set, events: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Every row of the given (untyped-death) cells removed, every other row kept in order; returns the cohort
    and its counts (rows, landmarks and cells removed and kept, per group; censored cells kept)."""
    m = data["track_id"].astype(str).isin(tracks).to_numpy()
    rem, keep = data[m], data[~m]
    lm_rem = rem.drop_duplicates(subset=KEY)
    lm_keep = keep.drop_duplicates(subset=KEY)
    counts = {
        "rows_removed": int(m.sum()), "rows_kept": int((~m).sum()),
        "landmarks_removed": int(len(lm_rem)), "landmarks_kept": int(len(lm_keep)),
        "tracks_removed": int(rem["track_id"].nunique()), "tracks_kept": int(keep["track_id"].nunique()),
        "removed_per_group": {str(g): {"landmarks": int(len(d)), "tracks": int(d["track_id"].nunique())}
                              for g, d in lm_rem.groupby("group")},
        "removed_tracks": sorted(rem["track_id"].astype(str).unique().tolist()),
    }
    if events is not None:
        cens = set(events.loc[events["event_type"] == "CENSORED", "track_id"].astype(str))
        kt = set(keep["track_id"].astype(str))
        counts["administratively_censored_tracks_kept"] = int(len(kt & cens))
        counts["administratively_censored_tracks_removed"] = int(len(set(rem["track_id"].astype(str)) & cens))
    return keep.reset_index(drop=True), counts


def end_of_followup_offset(lm: pd.DataFrame, bin_frames: int) -> np.ndarray:
    """Per row: the offset after t_obs of the last bin boundary inside the known follow-up of a landmark
    without a typed death, bin_frames * floor((t_end_true - t_obs) / bin_frames); inf for a typed death."""
    t = lm["t_obs_frame"].to_numpy(dtype=float)
    end = lm["t_end_true"].to_numpy(dtype=float)
    typed = lm["true_event"].isin(CAUSES).to_numpy() & np.isfinite(lm["t_event_true"].to_numpy(dtype=float))
    off = bin_frames * np.floor((end - t) / bin_frames)
    return np.where(typed, np.inf, off)


def corrected_censor_offset(tr: pd.DataFrame, bin_frames: int) -> np.ndarray:
    """The censor offset of the survival regime: min(injected censor offset, end-of-follow-up offset)."""
    return np.minimum(tr["censor_offset"].to_numpy(dtype=float), end_of_followup_offset(tr, bin_frames))


def at_risk_bins(base: pd.DataFrame, censor_offset: np.ndarray, bin_frames: int, n_bins: int) -> np.ndarray:
    """How many bins each landmark is at risk in under the survival head's rule: at risk in bin k when its event
    bin and its censor bin are both >= k, k = 1..n_bins (used for counts only)."""
    dt = (base["t_event_true"] - base["t_obs_frame"]).to_numpy(dtype=float)
    typed = base["true_event"].isin(CAUSES).to_numpy() & np.isfinite(dt) & (np.nan_to_num(dt, nan=0.0) > 0)
    with np.errstate(invalid="ignore"):
        ev_bin = np.where(typed, np.ceil(dt / bin_frames), np.inf)
    c = np.asarray(censor_offset, dtype=float)
    cens_bin = np.where(np.isfinite(c), np.ceil(c / bin_frames), np.inf)
    return np.clip(np.minimum(np.minimum(ev_bin, cens_bin), n_bins), 0, n_bins).astype(int)


def correction_counts(tr: pd.DataFrame, bin_frames: int, n_bins: int) -> dict:
    """What ending the risk set at the known end of follow-up removes from one training set (training landmarks
    only), with the injected-censor diagnostics."""
    base = tr.drop_duplicates(subset=KEY)
    orig = at_risk_bins(base, base["censor_offset"].to_numpy(dtype=float), bin_frames, n_bins)
    corr = at_risk_bins(base, corrected_censor_offset(base, bin_frames), bin_frames, n_bins)
    return {"training_landmarks": int(len(base)), "bins_at_risk_original": int(orig.sum()),
            "bins_at_risk_corrected": int(corr.sum()), "bins_removed": int((orig - corr).sum()),
            "landmarks_truncated": int((corr < orig).sum()),
            "unchanged_injected_censor_rule": injected_bin_diagnostics(base, corrected_censor_offset(base, bin_frames),
                                                                       bin_frames, n_bins)}


def injected_bin_diagnostics(base: pd.DataFrame, offset: np.ndarray, bin_frames: int, n_bins: int) -> dict:
    """Counts for the injected censors: the bin in which a censor falls stays at risk (ceil), so it is labelled
    survived, or with a typed death that comes later inside that bin. ``event_rows`` is the number of training event
    rows under ``offset``, the denominator of the share of such later deaths."""
    c = base["censor_offset"].to_numpy(dtype=float)
    dt = (base["t_event_true"] - base["t_obs_frame"]).to_numpy(dtype=float)
    typed = base["true_event"].isin(CAUSES).to_numpy() & np.isfinite(dt)
    n_risk = at_risk_bins(base, offset, bin_frames, n_bins)
    with np.errstate(invalid="ignore"):
        cb = np.ceil(c / bin_frames)
        ev_bin = np.where(typed, np.ceil(dt / bin_frames), np.inf)
        inside = (np.isfinite(c) & (cb >= 1) & (cb <= n_bins) & (np.mod(c, bin_frames) != 0) & (n_risk >= cb)
                  & ~(typed & (dt <= c)))
    later = inside & typed & (ev_bin == cb)
    event = typed & (np.nan_to_num(dt, nan=0.0) > 0) & (ev_bin <= n_risk)
    return {"censor_bin_at_risk": int(inside.sum()), "labelled_survived": int((inside & ~later).sum()),
            "labelled_with_a_later_death_in_that_bin": int(later.sum()), "event_rows": int(event.sum())}


def reach_at_rate0(data: pd.DataFrame, bin_frames: int, n_bins: int) -> dict:
    """With nothing injected, over every landmark of ``data``: the landmarks without a typed death whose bins the
    head would credit past their last full bin of known follow-up, and those bins."""
    base = data.drop_duplicates(subset=KEY)
    inf = np.full(len(base), np.inf)
    orig = at_risk_bins(base, inf, bin_frames, n_bins)
    corr = at_risk_bins(base, np.minimum(inf, end_of_followup_offset(base, bin_frames)), bin_frames, n_bins)
    return {"landmarks": int((corr < orig).sum()), "bins": int((orig - corr).sum())}


def run_unit(data: pd.DataFrame, cols: list, fold: dict, regime: str, seed: int, cfg: dict, cfg_w: dict,
             level: tuple = LEVEL, correct_d: bool = True) -> dict:
    """One unit (held-out group, regime, seed): censoring injected into the training landmarks, the regime trained
    and the held-out landmarks scored. ``data`` is the cohort (oracle rows joined to the history features). Returns
    y_<h>, p_<regime>_<cause>_<h> and, for the survival regime, its landmark index and per-horizon masks."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime}")
    mech, rate = level
    interval = float(cfg_w["frame_interval_min"])
    max_h_frames = int(max(cfg_w["horizons_min"]) / interval)
    eff_mech = "i_administrative" if rate == 0 else mech
    g = fold["test_group"]
    te = data[data["group"] == g].copy()
    tr = data[data["group"] != g].copy()
    if te.empty or tr.empty:
        raise ValueError(f"fold {fold['fold']}: an empty side")
    extra = [c for c in ("dry_mass_pg__last", "t_event_true") if c in tr]
    lm = tr.drop_duplicates(subset=KEY)[KEY + ["treatment"] + extra]
    co, achieved = inject_censor_time(lm, tr, rate, eff_mech, seed, max_h_frames, interval)
    lm = lm.assign(censor_offset=co)
    tr = tr.merge(lm[KEY + ["censor_offset"]], on=KEY, how="left")
    te = te.assign(censor_offset=np.inf)
    payload = {"group": g, "seed": seed, "rate": rate, "mechanism": eff_mech, "train_frac": 1.0,
               "n_train_movies": int(tr["movie_id"].nunique()), "regime": regime,
               "achieved_censoring_rate": achieved}
    hs = [int(h) for h in cfg_w["horizons_min"]]
    for h in hs:
        trh, teh = tr[tr["horizon_min"] == h], te[te["horizon_min"] == h]
        if trh.empty or teh.empty:
            continue
        payload[f"y_{h}"] = teh["oracle_label"].to_numpy()
        if regime in BINARY:
            hf = int(h / interval)
            for c in CAUSES:
                payload[f"p_{regime}_{c}_{h}"] = train_predict_horizon(trh, teh, cols, c, regime, hf, seed)
    if regime == "d_joint_survival":
        bin_f = int(cfg["survival_grid"]["bin_frames"])
        n_bins = int(cfg["survival_grid"]["n_bins"])
        payload["correction"] = correction_counts(tr, bin_f, n_bins)
        tr_d = tr.assign(censor_offset=corrected_censor_offset(tr, bin_f)) if correct_d else tr
        d_out = train_predict_survival(tr_d, te, cols, cfg, cfg_w, seed, interval_censored=False)
        for h in hs:
            if h in d_out:
                for c in CAUSES:
                    payload[f"p_{regime}_{c}_{h}"] = d_out[h][c]
        if "_index" in d_out:
            idx = d_out["_index"]
            payload["d_index_track"] = idx["track_id"].to_numpy().astype(str)
            payload["d_index_tobs"] = idx["t_obs_frame"].to_numpy()
            payload["d_index_movie"] = idx["movie_id"].to_numpy().astype(str)
            for h in hs:
                teh = te[te["horizon_min"] == h]
                kk = set(map(tuple, teh[KEY].to_numpy().tolist()))
                payload[f"d_mask_{h}"] = np.array([tuple(r) in kk for r in idx[KEY].to_numpy().tolist()], dtype=bool)
    return payload


def build_data(windows, events, movies, hist, cfg, cfg_w) -> tuple:
    """The oracle rows of the panel joined to the history features, then the cohort without the untyped-death
    cells: (cohort, feature columns, cohort counts)."""
    windows = windows[windows["panel"] == cfg["protocol"]["panel"]]
    windows = windows[~windows["movie_id"].isin(cfg["protocol"]["morphology_excluded_movies"])]
    oracle = build_oracle(windows, events, movies, cfg_w)
    cols = [c for c in hist.columns if c not in ("track_id", "frame_index")]
    data = oracle.merge(hist, left_on=["track_id", "t_obs_frame"], right_on=["track_id", "frame_index"], how="inner")
    cohort, counts = declared_cohort(data, untyped_death_tracks(events), events)
    counts["untyped_death_tracks_in_events"] = sorted(untyped_death_tracks(events))
    counts["untyped_death_tracks_without_rows"] = sorted(untyped_death_tracks(events) - set(counts["removed_tracks"]))
    counts["landmarks_by_group_kept"] = {str(g): int(len(d)) for g, d in
                                         cohort.drop_duplicates(subset=KEY).groupby("group")}
    counts["rows_by_horizon_kept"] = {str(int(h)): int(n) for h, n in cohort["horizon_min"].value_counts().items()}
    bf, nb = int(cfg["survival_grid"]["bin_frames"]), int(cfg["survival_grid"]["n_bins"])
    counts["correction_reach_rate0_cohort"] = reach_at_rate0(cohort, bf, nb)
    return cohort, cols, counts


# ------------------------------------------------------------------ the command
def set_threads(n: int) -> None:
    """LightGBM threads of the binary classifiers and of the survival head."""
    LGB_BINARY["num_threads"] = int(n)
    head.set_threads(n)


def unit_name(group: str, regime: str, seed: int) -> str:
    return f"{group}__{regime}__s{int(seed)}"


def unit_done(units: Path, u: str) -> bool:
    return (units / f"{u}.npz").is_file() and (units / f"{u}.json").is_file()


def write_unit(units: Path, u: str, payload: dict, meta: dict) -> None:
    """The npz, then the json (a killed run leaves no unit that counts as done)."""
    units.mkdir(parents=True, exist_ok=True)
    tmp = units / f".{u}.npz"
    np.savez_compressed(tmp, **{k: v for k, v in payload.items() if k != "correction"})
    tmp.replace(units / f"{u}.npz")
    tj = units / f".{u}.json"
    tj.write_text(json.dumps(meta, indent=1, default=float) + "\n")
    tj.replace(units / f"{u}.json")


def load_data(tables_dir: Path | None, history: Path | None, cfg: dict, cfg_w: dict) -> tuple:
    """The cohort, its feature columns and counts from the forecast-row, event, movie and cell tables."""
    from deadahead.features.history import history_features
    t = load_tables(tables_dir, names=("windows", "events", "movies"))
    if history is not None:
        hist = pd.read_parquet(history)
    else:
        cells = (Path(tables_dir) / "cells.parquet") if tables_dir is not None else None
        hist = history_features(int(cfg_w["landmark"]["history_frames_W"]), cells)
    return build_data(t["windows"], t["events"], t["movies"], hist, cfg, cfg_w)


def main(argv=None) -> int:
    from deadahead.survival.config import head_config, windows_config
    ap = argparse.ArgumentParser(prog="deadahead censoring-experiment", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--group", choices=GROUPS, action="append", help="a held-out group (repeatable)")
    who.add_argument("--all", action="store_true", help="every held-out group")
    ap.add_argument("--regime", choices=REGIMES, action="append", help="default: all four")
    ap.add_argument("--seed", type=int, action="append", help="default: the three seeds of the survival head")
    ap.add_argument("--tables-dir", type=Path, help="folder with windows, events, movies and cells .parquet")
    ap.add_argument("--history", type=Path, help="precomputed history features (.parquet); default: built and cached")
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="LightGBM threads (the released units used 1)")
    ap.add_argument("--out-dir", type=Path, help="default: runs/censoring")
    a = ap.parse_args(argv)
    cfg, cfg_w = head_config(), windows_config()
    set_threads(a.threads)
    groups = list(GROUPS) if a.all else a.group
    regimes = a.regime or list(REGIMES)
    seeds = a.seed or [int(s) for s in cfg["protocol"]["seeds"]]
    out = a.out_dir or runs_dir() / "censoring"
    units = out / "units"
    todo = [(g, r, s) for g in groups for r in regimes for s in seeds if not unit_done(units, unit_name(g, r, s))]
    if not todo:
        print(f"every requested unit is already in {units}")
        return 0
    t0 = time.time()
    data, cols, counts = load_data(a.tables_dir, a.history, cfg, cfg_w)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cohort.json").write_text(json.dumps(counts, indent=1) + "\n")
    print(f"data built in {time.time() - t0:.1f} s: {counts['rows_kept']:,} rows, {counts['landmarks_kept']:,} "
          f"landmarks; {counts['landmarks_removed']} landmarks of {counts['tracks_removed']} untyped-death cells "
          f"left out", flush=True)
    for g, r, s in todo:
        u = unit_name(g, r, s)
        t1 = time.time()
        payload = run_unit(data, cols, {"fold": GROUPS.index(g), "test_group": g}, r, s, cfg, cfg_w)
        meta = {"unit": u, "group": g, "regime": r, "seed": s,
                "level": {"mechanism": LEVEL[0], "rate": LEVEL[1]},
                "achieved_censoring_rate": float(payload["achieved_censoring_rate"]),
                "correction": payload.get("correction"), "seconds": round(time.time() - t1, 1),
                "lightgbm": lgb.__version__, "threads": int(LGB_BINARY["num_threads"])}
        write_unit(units, u, payload, meta)
        print(f"  {u}: {meta['seconds']} s, achieved censoring rate {meta['achieved_censoring_rate']:.3f}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
