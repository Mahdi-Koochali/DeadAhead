"""Warnings: the warning rows and their roles, training-only alarm thresholds, sustained coverage before death.

A threshold is set on the reference rows of the eight training groups (rows with no death in the next 240 min and
at least 240 min of follow-up; each row stands for 0.5 cell-hour) so that at most k = floor(alpha |reference|)
reference rows score at or above an alarmed row. A death is "alarmed at least X min ahead" when every 30-min bin
from (0, 30] to the bin ending at X holds an alarmed row of that cell.
"""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction

import numpy as np
import pandas as pd

from deadahead.evaluation.aggregate import LEFT_OUT_FOLD, both_folds
from deadahead.evaluation.rows import CAUSES, GROUPS, KEY, LABEL_UNTYPED, apply_cohort, key_frame

CAUSE_IX = {"APOPTOSIS": 1, "LYTIC": 2}
FRAME_MIN = 3.0                              # minutes between frames
CUT_FRAME = 438                              # frames 0..438 are scored
FOLLOW_MIN = 240                             # reference rows: no death and followed through the next 240 min
LANDMARK_CUT = 358                           # rows t <= 358 have 240 min of follow-up inside the cut
ROW_HOURS = 0.5                              # each reference row is 0.5 cell-hour
NOMINAL = {"0.2_per_cell_hour": Fraction(1, 10), "0.4_per_cell_hour": Fraction(1, 5)}
PRIMARY_RATE = "0.2_per_cell_hour"
WARNING_HORIZON = 30                         # the warning score is the 30-min cumulative incidence F_c(30)
SUSTAINED_X = (30, 60, 90, 120, 180, 240)
TIE_SEED = 2601562                           # tie-break of the elapsed-time score


# ------------------------------------------------------------------ outcomes and roles of the warning rows
def oracle_outcomes(lm: pd.DataFrame, events: pd.DataFrame, movies: pd.DataFrame,
                    untyped_as_event: bool = False) -> pd.DataFrame:
    """Per row, the complete-movie outcome: a typed death (APOPTOSIS or LYTIC with a t_event) at T = (t_event - t_obs)
    x 3 min, anything else alive to the movie's end; C = (num_frames - 1 - t_obs) x 3 min. Also the death frame and
    the untyped-death flag. With ``untyped_as_event`` an untyped death is status 3 at its own frame."""
    ev = events.set_index("track_id")
    nf = dict(zip(movies["movie_id"], movies["num_frames"]))
    t_obs = lm["t_obs_frame"].to_numpy(float)
    etype = lm["track_id"].map(ev["event_type"]).fillna("").to_numpy()
    tev = lm["track_id"].map(ev["t_event"]).to_numpy(float)
    last = lm["movie_id"].map(nf).to_numpy(float) - 1
    typed = np.isin(etype, CAUSES) & np.isfinite(tev)
    untyped = (etype == LABEL_UNTYPED)
    C = (last - t_obs) * FRAME_MIN
    T = np.where(typed, (tev - t_obs) * FRAME_MIN, C)
    delta = np.where(typed, np.select([etype == "APOPTOSIS", etype == "LYTIC"], [1, 2], 0), 0)
    if untyped_as_event:
        ut = untyped & np.isfinite(tev)
        T = np.where(ut, (tev - t_obs) * FRAME_MIN, T)
        delta = np.where(ut, 3, delta)
    out = lm[KEY + (["group"] if "group" in lm else [])].copy()
    out["T_min"], out["delta"], out["C_min"] = T.astype(float), delta.astype(int), C.astype(float)
    out["death_frame"] = np.where(typed, tev, np.nan)
    out["untyped_track"] = untyped
    return out.reset_index(drop=True)


def row_roles(T, delta, C, death_frame, cut: int = CUT_FRAME, follow: float = FOLLOW_MIN) -> dict:
    """The roles of each row: event rows (a typed death at frame <= 438; lead = T), reference rows (no typed death in
    the next 240 min and followed >= 240 min), the warning window (a typed death within 240 min: neither) and the
    censoring bucket (followed < 240 min without a death: counted, never a reference row)."""
    T, delta, C = np.asarray(T, float), np.asarray(delta, int), np.asarray(C, float)
    typed = np.isin(delta, (1, 2))
    warn = typed & (T <= follow)
    negative = (~warn) & ((typed & (T > follow)) | ((delta == 3) & (T > follow)) | ((delta == 0) & (C >= follow)))
    bucket = (~warn) & (~negative) & (delta == 0)
    df = np.asarray(death_frame, float)
    event = typed & np.isfinite(df) & (df <= cut)
    return {"event_cause": np.where(event, delta, 0), "lead_min": np.where(event, T, np.nan), "negative": negative,
            "warning_window": warn, "censoring_bucket": bucket}


def truncate_at_cut(df: pd.DataFrame, cut: int = CUT_FRAME) -> pd.DataFrame:
    """Labels with follow-up truncated at frame ``cut``: C = min(C, (cut - t) x 3 min); a death after the cut becomes
    censoring at the cut; ``death_frame`` beyond the cut is dropped."""
    out = df.copy()
    t = out["t_obs_frame"].to_numpy(float)
    c_cut = (cut - t) * FRAME_MIN
    Cm = np.minimum(out["C_min"].to_numpy(float), c_cut)
    T, d = out["T_min"].to_numpy(float), out["delta"].to_numpy(int)
    late = (d != 0) & (T > c_cut)
    out["C_min"] = Cm
    out["T_min"] = np.where(late | (d == 0), Cm, T)
    out["delta"] = np.where(late, 0, d)
    if "death_frame" in out:
        dfr = out["death_frame"].to_numpy(float)
        out["death_frame"] = np.where(np.isfinite(dfr) & (dfr <= cut), dfr, np.nan)
    return out


def warning_rows(row_keys: pd.DataFrame, events: pd.DataFrame, movies: pd.DataFrame, untyped: set,
                 common_landmark: bool = False) -> tuple[pd.DataFrame, dict]:
    """The warning rows (the 30-min rows with a determined label), the cohort applied, with complete-movie outcomes
    and the roles of ``row_roles`` (``negative`` marks the reference rows). With ``common_landmark``: rows
    t <= 358, follow-up truncated at frame 438, roles recomputed."""
    kept, removed = apply_cohort(row_keys[KEY + ["group"]], untyped)
    o = oracle_outcomes(kept, events, movies)
    if o["untyped_track"].any():     # the outcome table codes an untyped death as alive
        raise AssertionError("an untyped-death cell is still in the warning rows after the cohort")
    info = {"cohort_removed": removed}
    if common_landmark:
        n0 = len(o)
        o = truncate_at_cut(o[o["t_obs_frame"] <= LANDMARK_CUT].reset_index(drop=True))
        info["common_landmark"] = {"t_max": LANDMARK_CUT, "rows_before": int(n0), "rows_after": len(o)}
    roles = row_roles(o["T_min"], o["delta"], o["C_min"], o["death_frame"])
    rows = o[KEY + ["group"]].copy()
    for k, v in roles.items():
        rows[k] = v
    return rows.sort_values(KEY).reset_index(drop=True), info


def rows_id(df: pd.DataFrame) -> dict:
    """The row identifiers of a population as a count and the sha256 of its sorted keys."""
    k = key_frame(df).sort_values(KEY).reset_index(drop=True)
    h = hashlib.sha256("\n".join(f"{m}|{t}|{f}" for m, t, f in k.itertuples(index=False)).encode()).hexdigest()
    return {"n": len(k), "keys_sha256": h}


def reference_record(rows: pd.DataFrame, ref_mask, g: str, removed: dict | None = None) -> dict:
    """The identifiers and exclusion counts of the training reference rows of outer group g: per training group the
    reference rows (count, keys' sha256) and the rows outside it by reason."""
    m = np.asarray(ref_mask, bool)
    tr = rows["group"].to_numpy() != g
    out = {"outer_group": g, "pooled": rows_id(rows[m & tr]), "per_training_group": {}}
    for h in [x for x in GROUPS if x != g]:
        mh = rows["group"].to_numpy() == h
        r = {"reference": rows_id(rows[m & mh]), "rows": int(mh.sum()), "outside_reference": int((mh & ~m).sum())}
        for col in ("warning_window", "censoring_bucket"):
            if col in rows:
                r[f"outside_{col}"] = int((mh & rows[col].to_numpy(bool)).sum())
        if removed:
            r["cohort_removed"] = (removed.get("per_group") or {}).get(h, {"landmarks": 0, "tracks": 0})
        out["per_training_group"][h] = r
    return out


# ------------------------------------------------------------------ thresholds
def budget_k(n_neg: int, alpha: Fraction) -> int:
    """k = floor(alpha |N_g|), in integer arithmetic."""
    return math.floor(Fraction(alpha) * int(n_neg))


def alarmed(score, negative, alpha: Fraction) -> tuple[np.ndarray, dict]:
    """A row is alarmed iff the number of reference rows with score >= its score is <= k (ties count against the
    alarm). Returns the flags and the threshold record (k, |N|, realised rate, a tie straddling the cut)."""
    s, neg = np.asarray(score, float), np.asarray(negative, bool)
    if not np.all(np.isfinite(s)):
        raise ValueError("a non-finite score")
    ns = np.sort(s[neg])
    k = budget_k(ns.size, alpha)
    n_ge = ns.size - np.searchsorted(ns, s, side="left")
    flags = n_ge <= k
    realised = float(flags[neg].mean()) if neg.any() else float("nan")
    # a tie straddling the cut: the k-th and the (k+1)-th largest reference scores are equal, so the whole tie
    # stays unalarmed; a tie wholly below the cut changes nothing and is not recorded
    cut_tie = None
    if 1 <= k < ns.size and ns[ns.size - k] == ns[ns.size - k - 1]:
        v = ns[ns.size - k - 1]
        cut_tie = {"score": float(v), "n_negatives_tied": int((ns == v).sum())}
    return flags, {"k": int(k), "n_negatives": int(ns.size), "alpha": str(alpha), "realised_rate": realised,
                   "cut_tie": cut_tie}


def threshold_and_apply(ref_scores, test_scores, alpha: Fraction) -> tuple[np.ndarray, dict]:
    """The training-only threshold: the pooled reference rows of the 8 training groups set k = floor(alpha |ref|),
    and a held-out row is alarmed iff at most k reference rows score >= it. Returns the held-out flags and the
    threshold record (k, |ref|, the training rate as a row fraction and per cell-hour)."""
    ref, te = np.asarray(ref_scores, float), np.asarray(test_scores, float)
    if ref.size == 0:
        raise ValueError("an empty reference population")
    flags, thr = alarmed(np.r_[ref, te], np.r_[np.ones(ref.size, bool), np.zeros(te.size, bool)], alpha)
    ns = np.sort(ref)
    k = thr["k"]
    # the lowest score that is alarmed: the k-th largest reference score is alarmed only when no tie straddles it
    thr["score_threshold_note"] = "alarmed iff #(reference >= score) <= k"
    thr["kth_largest_reference"] = float(ns[ns.size - k]) if k >= 1 else None
    thr["training_rate_per_cell_hour"] = thr["realised_rate"] / ROW_HOURS
    thr["nominal_per_cell_hour"] = float(alpha) / ROW_HOURS
    return flags[ref.size:], thr


def time_in_movie_score(t_obs_frame, seed: int = TIE_SEED, tie_break: bool = True) -> np.ndarray:
    """Score = the row's frame (later ranks higher). With ``tie_break`` ties are broken by a uniform jitter in [0, 1)
    from a fixed seed, which keeps every between-frame order; without it the frame itself."""
    t = np.asarray(t_obs_frame, dtype=float)
    if not tie_break:
        return t
    if np.any(t != np.round(t)):
        raise ValueError("t_obs_frame must be whole frames")
    return t + np.random.default_rng(seed).random(t.size)


def time_scores(rows: pd.DataFrame) -> pd.Series:
    """The elapsed-time warning score: drawn once over ``rows`` sorted by KEY, so a row keeps one score whether it
    is a training reference row or a held-out row."""
    k = key_frame(rows)
    order = k.sort_values(KEY, kind="mergesort").index.to_numpy()
    s = np.empty(len(k))
    s[order] = time_in_movie_score(k["t_obs_frame"].to_numpy()[order])
    return pd.Series(s, index=rows.index)


# ------------------------------------------------------------------ coverage before death
def in_bin(lead, lo: float, hi: float) -> np.ndarray:
    lead = np.asarray(lead, dtype=float)
    return np.isfinite(lead) & (lead > lo) & (lead <= hi)


def _event_bins(track, lead_min, event_cause, cause: int) -> dict:
    """{track: {bin index 0..7: row}}; at most one row per bin per event (rows are 30 min apart)."""
    ev = (np.asarray(event_cause) == cause) & in_bin(lead_min, 0, 240)
    lead = np.asarray(lead_min, float)
    trk = np.asarray(track).astype(str)
    out: dict = {}
    for i in np.flatnonzero(ev):
        b = int(math.ceil(lead[i] / 30.0)) - 1
        d = out.setdefault(trk[i], {})
        if b in d:
            raise ValueError(f"event {trk[i]} has two landmarks in lead bin {b} (landmarks are 30 min apart)")
        d[b] = int(i)
    return out


def sustained(flags, track, lead_min, event_cause, cause: int, alpha: Fraction) -> dict:
    """"Alarmed at least X min ahead": every bin from (0,30] to the bin ending at X alarmed; the share over
    eligible events (a row in each of those bins), with the chance bounds alpha^m and alpha."""
    flags = np.asarray(flags, bool)
    eb = _event_bins(track, lead_min, event_cause, cause)
    a = float(alpha)
    out = {}
    for X in SUSTAINED_X:
        m = X // 30
        elig = [d for d in eb.values() if all(b in d for b in range(m))]
        hit = sum(all(flags[d[b]] for b in range(m)) for d in elig)
        out[str(X)] = {"n_eligible_events": len(elig), "n_alarmed": int(hit),
                       "share": hit / len(elig) if elig else float("nan"),
                       "chance_lower": a ** m, "chance_upper": a}
    return out


def median_sustained_warning(flags, track, lead_min, event_cause, cause: int) -> dict:
    """Per event with a row in all eight bins: the largest X in {30, 60, 90, 120, 180, 240} for which it is alarmed
    at least X min ahead (0 if its (0,30] row is not alarmed); the median over those events, with the count left
    out."""
    flags = np.asarray(flags, bool)
    eb = _event_bins(track, lead_min, event_cause, cause)
    full = [d for d in eb.values() if all(b in d for b in range(8))]
    vals = []
    for d in full:
        run = 0
        while run < 8 and flags[d[run]]:
            run += 1
        vals.append(max([X for X in SUSTAINED_X if X <= 30 * run], default=0))
    return {"n_events_all_eight_bins": len(full), "n_events_left_out": len(eb) - len(full),
            "median_min": float(np.median(vals)) if vals else float("nan")}


# ------------------------------------------------------------------ rates and summaries
def achieved_rate(flags, reference) -> dict:
    f, r = np.asarray(flags, bool), np.asarray(reference, bool)
    n = int(r.sum())
    a = int((f & r).sum())
    return {"alarmed_reference_rows": a, "reference_rows": n, "exposure_cell_hours": n * ROW_HOURS,
            "rate_per_cell_hour": a / (n * ROW_HOURS) if n else float("nan")}


def pooled_rate(per_group_seed_counts: dict, groups=GROUPS) -> dict:
    """Exposure-weighted over the held-out groups' reference rows, per seed, then the mean over seeds."""
    seeds = sorted({s for g in groups if g in per_group_seed_counts for s in per_group_seed_counts[g]})
    per_seed = {}
    for s in seeds:
        a = sum(per_group_seed_counts[g][s]["alarmed_reference_rows"] for g in groups if g in per_group_seed_counts
                and s in per_group_seed_counts[g])
        n = sum(per_group_seed_counts[g][s]["reference_rows"] for g in groups if g in per_group_seed_counts
                and s in per_group_seed_counts[g])
        per_seed[s] = a / (n * ROW_HOURS) if n else float("nan")
    return {"rate_per_cell_hour": float(np.mean(list(per_seed.values()))) if per_seed else float("nan"),
            "per_seed": per_seed}


def warning_unit(flags, rows_g: pd.DataFrame, cause: str, alpha: Fraction) -> dict:
    """One (model, group, seed, cause, rate) on the held-out group's warning rows: the achieved rate on the
    reference rows, sustained coverage per X with eligible deaths, the median warning."""
    lead, ev = rows_g["lead_min"].to_numpy(), rows_g["event_cause"].to_numpy()
    trk = rows_g["track_id"].astype(str).to_numpy()
    c = CAUSE_IX[cause]
    return {"achieved": achieved_rate(flags, rows_g["negative"].to_numpy(bool)),
            "sustained": sustained(flags, trk, lead, ev, c, alpha),
            "median_warning": median_sustained_warning(flags, trk, lead, ev, c)}


def warning_summary(units: dict) -> dict:
    """{(g, s): warning_unit} -> the achieved rate (equal-group macro and exposure-weighted, 9 and 8 groups), the
    sustained shares per X with eligible deaths per group, the median warning: seed means, then macros."""
    gs = sorted({g for g, _ in units})
    rate_pgs = {g: {s: u["achieved"]["rate_per_cell_hour"] for (gg, s), u in units.items() if gg == g} for g in gs}
    counts = {g: {s: u["achieved"] for (gg, s), u in units.items() if gg == g} for g in gs}
    eight = tuple(g for g in GROUPS if g != LEFT_OUT_FOLD)
    out = {"achieved_rate_equal_group_macro": both_folds(rate_pgs),
           "achieved_rate_exposure_weighted": {"nine_folds": pooled_rate(counts, GROUPS),
                                               "eight_folds_without_PNT1A_bp": pooled_rate(counts, eight)},
           "sustained": {}, "median_warning_min": None}
    xs = sorted({X for u in units.values() for X in u["sustained"]}, key=int)
    for X in xs:
        pgs = {g: {s: u["sustained"][X]["share"] for (gg, s), u in units.items() if gg == g} for g in rate_pgs}
        elig = {g: next(u["sustained"][X]["n_eligible_events"] for (gg, s), u in units.items() if gg == g)
                for g in rate_pgs}
        first = next(iter(units.values()))["sustained"][X]
        out["sustained"][X] = {**both_folds(pgs), "eligible_deaths_per_group": elig,
                               "chance_lower": first["chance_lower"], "chance_upper": first["chance_upper"]}
    med = {g: {s: u["median_warning"]["median_min"] for (gg, s), u in units.items() if gg == g} for g in rate_pgs}
    out["median_warning_min"] = {**both_folds(med), "events_all_eight_bins_per_group": {
        g: next(u["median_warning"]["n_events_all_eight_bins"] for (gg, s), u in units.items() if gg == g)
        for g in rate_pgs}}
    return out
