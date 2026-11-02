"""Survival metrics of one held-out group at fixed horizons, numpy only: IPCW Brier score and IPA, Wolbers'
concordance, expected over observed incidence and the IPCW calibration intercept and slope.

One landmark row carries T (minutes from the landmark to the death, else to the end of observation), delta (0
censored, 1 apoptosis, 2 lytic, 3 untyped) and C (the known censoring time in minutes, the end of the movie). For cause
c and horizon t: Y(t) = 1 iff delta == c and T <= t; a row is determined at t iff (delta != 0 and T <= t) or C >= t.
Every metric is cause-specific and computed on one held-out group's rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from deadahead.baselines.elapsed_time import _fit_logistic

CENSORED, APOPTOSIS, LYTIC, UNTYPED = 0, 1, 2, 3
CAUSES = ("APOPTOSIS", "LYTIC")
CAUSE_IX = {"APOPTOSIS": APOPTOSIS, "LYTIC": LYTIC}
HORIZONS_MIN = (30, 60, 120, 240)
POSITIVITY_MIN_G = 0.25                              # a horizon with G(t-) below this is not scorable
OBSERVED_CAPTION = "the observed incidence is the Aalen-Johansen estimate on the rows that carry a prediction"
PAIR_SETS = ("no_same_track", "all_pairs", "cross_movie")   # concordance pairs; the first is the primary one
IPCW_KEYS = ("brier_ipcw", "ipa", "ipa_training_null", "ipcw_intercept", "ipcw_slope")


# ------------------------------------------------------------------ outcome and determinedness
def _arr(x, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def outcome(T, delta, cause: int, t: float) -> np.ndarray:
    """Y_i(t) = 1 iff delta_i == cause and T_i <= t (an observed death of that cause by t)."""
    T, delta = _arr(T), _arr(delta, int)
    return ((delta == cause) & (T <= t)).astype(int)


def determined(T, delta, C, t: float) -> np.ndarray:
    """(delta_i != 0 and T_i <= t) or C_i >= t."""
    T, delta, C = _arr(T), _arr(delta, int), _arr(C)
    return ((delta != CENSORED) & (T <= t)) | (C >= t)


# ------------------------------------------------------------------ the censoring survivor G
class CensoringSurvivor:
    """G(u-) = P(C >= u): the empirical survivor function of the known censoring times of one population (every
    landmark's censoring time is known, so no reverse Kaplan-Meier is needed)."""

    def __init__(self, C):
        c = np.sort(_arr(C))
        if c.size == 0 or not np.all(np.isfinite(c)):
            raise ValueError("G needs a non-empty population of finite censoring times")
        self._c, self.n = c, int(c.size)

    def left(self, u) -> np.ndarray:
        """G(u-) = #{C_i >= u} / n, vectorised."""
        u = _arr(u)
        return (self.n - np.searchsorted(self._c, u, side="left")) / self.n

    def min_on(self, t_max: float = 240.0) -> float:
        """min_{t <= t_max} G(t-) = G(t_max-), G being non-increasing."""
        return float(self.left(t_max))

    def n_below(self, t_max: float = 240.0) -> int:
        """The number of landmarks with C_i < t_max."""
        return int(np.searchsorted(self._c, t_max, side="left"))


class NoWeights:
    """G == 1: the unweighted complete-case version of an IPCW estimator."""

    def left(self, u) -> np.ndarray:
        return np.ones(np.shape(np.asarray(u, dtype=float)))


def positivity(G: CensoringSurvivor, horizons=HORIZONS_MIN, threshold: float = POSITIVITY_MIN_G) -> dict:
    """A horizon t is not scorable when min_{u <= t} G(u-) = G(t-) < threshold. Returns min G on [0, 240], the count
    of C < 240, G at each horizon and the horizons that are not scorable."""
    per_h = {int(t): float(G.left(t)) for t in horizons}
    return {"min_G_0_240": G.min_on(240.0), "n_C_below_240": G.n_below(240.0), "G_at_horizon": per_h,
            "not_scorable_horizons": [t for t, g in per_h.items() if g < threshold], "threshold": threshold}


def ipcw_weights(T, delta, C, t: float, G: CensoringSurvivor) -> np.ndarray:
    """W_i(t) = 1{delta != 0, T <= t} / G(T_i-) + 1{no observed death by t, C_i >= t} / G(t-); 0 when not
    determined at t. A death of the other cause before t is in the first term (an observed outcome)."""
    T, delta, C = _arr(T), _arr(delta, int), _arr(C)
    dead = (delta != CENSORED) & (T <= t)
    seen = (~dead) & (C >= t)
    w = np.zeros(T.size)
    g_dead = G.left(T[dead])
    g_t = float(G.left(t))
    if (g_dead <= 0).any() or (seen.any() and g_t <= 0):
        raise ValueError("a weighted row has G = 0 (positivity fails)")
    w[dead] = 1.0 / g_dead
    w[seen] = 1.0 / g_t
    return w


def weight_sum_ratio(T, delta, C, t: float, G: CensoringSurvivor) -> float:
    """sum_i W_i(t) / N, a diagnostic of the weights (the Brier score divides by N)."""
    w = ipcw_weights(T, delta, C, t, G)
    return float(w.sum() / w.size)


# ------------------------------------------------------------------ Brier score, IBS, IPA
def _pred_on(pred, mask: np.ndarray) -> np.ndarray:
    p = _arr(pred)
    if p.shape != mask.shape:
        raise ValueError(f"prediction length {p.shape} differs from the population's {mask.shape}")
    if not np.all(np.isfinite(p[mask])):
        raise ValueError("a determined row has no finite prediction")
    return p


def brier_ipcw(T, delta, C, pred, cause: int, t: float, G: CensoringSurvivor) -> float:
    """BS_c(t) = (1/N) sum_i W_i(t) (Y_i(t) - F_c(t|x_i))^2 over the whole population (N = all landmarks; the
    undetermined ones carry weight 0 and need no prediction; Graf et al. 1999, competing-risks form)."""
    w = ipcw_weights(T, delta, C, t, G)
    m = w > 0
    p = _pred_on(pred, m)
    y = outcome(T, delta, cause, t)
    return float(np.sum(w[m] * (y[m] - p[m]) ** 2) / w.size)


def brier_complete_case(T, delta, C, pred, cause: int, t: float) -> float:
    """The unweighted complete-case Brier score on the determined rows."""
    m = determined(T, delta, C, t)
    if not m.any():
        return float("nan")
    p = _pred_on(pred, m)
    y = outcome(T, delta, cause, t)
    return float(np.mean((y[m] - p[m]) ** 2))


def trapezoid_weights(times) -> np.ndarray:
    """The trapezoid rule's weights over ``times``, normalised by the interval length: for (30, 60, 120, 240) this is
    (15, 45, 90, 60) / 210."""
    t = _arr(times)
    if t.size < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("the IBS needs at least two increasing times")
    w = np.zeros(t.size)
    d = np.diff(t)
    w[:-1] += d / 2.0
    w[1:] += d / 2.0
    return w / (t[-1] - t[0])


def ibs(bs_by_t: dict) -> float:
    """The integrated Brier score over [first, last] time by the trapezoid rule on the given horizons."""
    ts = sorted(bs_by_t)
    vals = _arr([bs_by_t[t] for t in ts])
    if not np.all(np.isfinite(vals)):
        return float("nan")
    return float(np.dot(trapezoid_weights(ts), vals))


def ipa(bs: float, bs_null: float) -> float:
    """IPA = 1 - BS / BS_null (Kattan and Gerds 2018); nan when the null Brier score is 0 or not finite."""
    if not np.isfinite(bs) or not np.isfinite(bs_null) or bs_null <= 0:
        return float("nan")
    return float(1.0 - bs / bs_null)


# ------------------------------------------------------------------ Aalen-Johansen
def aalen_johansen(T, delta, t: float, causes=(APOPTOSIS, LYTIC), w=None) -> dict:
    """Cumulative incidence of each cause by t with competing risks, optionally with row multiplicities ``w``:
    CIF_c(t) = sum_{u <= t} S(u-) d_c(u) / n(u), S the all-cause Kaplan-Meier (every status > 0 is an event)."""
    T, delta = _arr(T), _arr(delta, int)
    w = np.ones(T.size) if w is None else _arr(w)
    ev = (delta > CENSORED) & (T <= t) & (w > 0)
    if not ev.any():
        return {c: 0.0 for c in causes}
    ts = np.unique(T[ev])
    order = np.argsort(T, kind="stable")
    Ts, ws = T[order], w[order]
    cum = np.concatenate([[0.0], np.cumsum(ws)])
    n_risk = cum[-1] - cum[np.searchsorted(Ts, ts, side="left")]
    pos = np.searchsorted(ts, T[ev])
    d_all = np.bincount(pos, weights=w[ev], minlength=ts.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        haz = np.where(n_risk > 0, d_all / n_risk, 0.0)
    s_before = np.concatenate([[1.0], np.cumprod(1.0 - haz)[:-1]])
    out = {}
    for c in causes:
        evc = ev & (delta == c)
        d_c = np.bincount(np.searchsorted(ts, T[evc]), weights=w[evc], minlength=ts.size)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[c] = float(np.sum(np.where(n_risk > 0, s_before * d_c / n_risk, 0.0)))
    return out


def null_brier(T, delta, C, cause: int, t: float, G: CensoringSurvivor, T_ref=None, delta_ref=None) -> dict:
    """The null model: a constant prediction, the Aalen-Johansen incidence of the cause at t on a reference
    population (the held-out group's own by default), scored with the same IPCW Brier score."""
    Tr = T if T_ref is None else T_ref
    dr = delta if delta_ref is None else delta_ref
    f0 = aalen_johansen(Tr, dr, t, causes=(cause,))[cause]
    pred = np.full(_arr(T).size, f0)
    return {"null_prediction": f0, "bs_null": brier_ipcw(T, delta, C, pred, cause, t, G)}


# ------------------------------------------------------------------ Wolbers' concordance on determined rows
def concordance(T, delta, C, marker, cause: int, tau: float, G: CensoringSurvivor, track=None, movie=None,
                pair_set: str = "no_same_track", chunk: int = 512) -> dict:
    """Wolbers' C_c(tau) = P(M_i > M_j | delta_i = c, T_i <= tau, j free of c at T_i), on pairs whose members are
    both determined at tau, each pair weighted 1 / (G(T_i-) G(min(T_j, tau)-)).

    A pair (i, j) is comparable when i is a case (delta_i = c, T_i <= tau), j != i, and j has a competing status, or
    T_j > T_i, or j is censored at T_j = C_j = T_i. Two deaths of cause c at one time are not comparable; a marker
    tie counts 1/2. ``pair_set``: "no_same_track" (pairs from different cells), "all_pairs" or "cross_movie"."""
    T, delta, C, M = _arr(T), _arr(delta, int), _arr(C), _arr(marker)
    if pair_set not in PAIR_SETS:
        raise ValueError(pair_set)
    det = determined(T, delta, C, tau)
    if not np.all(np.isfinite(M[det])):
        raise ValueError("a determined row has no finite marker")
    cases = np.flatnonzero(det & (delta == cause) & (T <= tau))
    ctrl = np.flatnonzero(det)
    if pair_set == "no_same_track" and track is None:
        raise ValueError("no_same_track needs track ids")
    if pair_set == "cross_movie" and movie is None:
        raise ValueError("cross_movie needs movie ids")
    trk = None if track is None else np.asarray(track).astype(str)
    mov = None if movie is None else np.asarray(movie).astype(str)
    Tj, dj, Mj = T[ctrl], delta[ctrl], M[ctrl]
    gj = G.left(np.minimum(Tj, tau))
    num = den = 0.0
    n_pairs = 0
    for s in range(0, cases.size, chunk):
        ci = cases[s:s + chunk]
        Ti, Mi = T[ci][:, None], M[ci][:, None]
        comp = ((dj[None, :] != CENSORED) & (dj[None, :] != cause)) | (Tj[None, :] > Ti) \
            | ((dj[None, :] == CENSORED) & (Tj[None, :] == Ti))
        comp &= ci[:, None] != ctrl[None, :]
        if pair_set == "no_same_track":
            comp &= trk[ci][:, None] != trk[ctrl][None, :]
        elif pair_set == "cross_movie":
            comp &= mov[ci][:, None] != mov[ctrl][None, :]
        gi = G.left(T[ci])
        if (gi <= 0).any() or (gj[comp.any(axis=0)] <= 0).any():
            raise ValueError("a comparable pair has G = 0 (positivity fails)")
        w = comp / (gi[:, None] * np.where(gj > 0, gj, np.inf)[None, :])
        q = (Mi > Mj[None, :]) + 0.5 * (Mi == Mj[None, :])
        num += float(np.sum(w * q))
        den += float(np.sum(w))
        n_pairs += int(comp.sum())
    return {"c_index": (num / den) if den > 0 else float("nan"), "n_pairs": n_pairs, "n_cases": int(cases.size),
            "weight_sum": den, "pair_set": pair_set}


# ------------------------------------------------------------------ calibration
def expected_over_observed(pred, T, delta, cause: int, t: float) -> float:
    """E/O = mean prediction / Aalen-Johansen observed incidence at t on the same rows."""
    o = aalen_johansen(T, delta, t, causes=(cause,))[cause]
    return float(np.mean(_arr(pred)) / o) if o > 0 else float("nan")


def ipcw_intercept_slope(T, delta, C, pred, cause: int, t: float, G: CensoringSurvivor) -> tuple[float, float]:
    """The logistic fit y ~ a + b logit(p) with the weights W_i(t) on the determined rows (p clipped to
    [1e-6, 1 - 1e-6]); returns (a, b)."""
    w = ipcw_weights(T, delta, C, t, G)
    m = w > 0
    p = np.clip(_pred_on(pred, m)[m], 1e-6, 1 - 1e-6)
    y = outcome(T, delta, cause, t)[m].astype(float)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    x = np.log(p / (1 - p))
    beta = _fit_logistic(np.column_stack([np.ones_like(x), x]), y, w=w[m])
    return float(beta[0]), float(beta[1])


def calibration_intercept_slope(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration y ~ a + b logit(p), unweighted; perfect calibration is intercept 0, slope 1."""
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(prob / (1 - prob))
    y = np.asarray(y_true, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    beta = _fit_logistic(np.column_stack([np.ones_like(x), x]), y)
    return float(beta[0]), float(beta[1])


# ------------------------------------------------------------------ one held-out group and one fit
def metrics_block(pop_g: pd.DataFrame, preds: dict, horizons, train: pd.DataFrame | None, ibs_too: bool) -> dict:
    """Every metric of one held-out group and one fit: per cause and horizon the IPCW Brier score and its
    complete-case value, IPA against the held-out and the training Aalen-Johansen nulls, Wolbers' C (three pair
    sets and unweighted), E/O, the IPCW and complete-case intercept and slope; with ``ibs_too`` the IBS and its IPA.
    ``preds``: {h: {cause: array over pop_g}}. Values only; the scorability rules are applied by ``_scorable``."""
    T, d, C = (pop_g[k].to_numpy() for k in ("T_min", "delta", "C_min"))
    trk, mov = pop_g["track_id"].astype(str).to_numpy(), pop_g["movie_id"].astype(str).to_numpy()
    G = CensoringSurvivor(C)
    out = {"G": positivity(G, horizons), "n_population": int(len(pop_g)), "causes": {},
           "observed_caption": OBSERVED_CAPTION}
    out["G"]["weight_sum_ratio"] = {h: weight_sum_ratio(T, d, C, h, G) for h in horizons}
    for cname in CAUSES:
        c = CAUSE_IX[cname]
        per_h = {}
        for h in horizons:
            p = preds[h][cname]
            fin = np.isfinite(p)
            y = outcome(T, d, c, h)
            # events are typed deaths (one per cell); a death counts at every landmark within h of it
            r = {"n_determined": int(determined(T, d, C, h).sum()), "n_event_landmarks": int(y.sum()),
                 "n_events_tracks": int(np.unique(trk[y == 1]).size), "n_rows_predicted": int(fin.sum())}
            r["brier_ipcw"] = brier_ipcw(T, d, C, p, c, h, G)
            r["brier_complete_case"] = brier_complete_case(T, d, C, p, c, h)
            nb = null_brier(T, d, C, c, h, G)
            r["null_own"] = nb
            r["ipa"] = ipa(r["brier_ipcw"], nb["bs_null"])
            if train is not None:
                nt = null_brier(T, d, C, c, h, G, T_ref=train["T_min"].to_numpy(), delta_ref=train["delta"].to_numpy())
                r["null_training"] = {**nt, "training_undetermined": int((~determined(
                    train["T_min"], train["delta"], train["C_min"], h)).sum()),
                    "flag": "the null incidence estimated on the training groups"}
                r["ipa_training_null"] = ipa(r["brier_ipcw"], nt["bs_null"])
            r["c"] = {ps: concordance(T, d, C, p, c, h, G, track=trk, movie=mov, pair_set=ps)
                      for ps in PAIR_SETS}
            # the unweighted complete-case values beside the IPCW ones
            r["c"]["unweighted"] = concordance(T, d, C, p, c, h, NoWeights(), track=trk, movie=mov,
                                               pair_set="no_same_track")
            det = determined(T, d, C, h)
            # the complete-case IPA's null: the complete-case prevalence of the cause among the determined rows
            f0_cc = float(y[det].mean()) if det.any() else float("nan")
            r["null_complete_case"] = f0_cc
            bs_null_cc = brier_complete_case(T, d, C, np.full(len(T), f0_cc), c, h)
            r["ipa_complete_case"] = ipa(r["brier_complete_case"], bs_null_cc)
            r["cc_intercept"], r["cc_slope"] = calibration_intercept_slope(
                outcome(T, d, c, h)[det].astype(float), p[det])
            r["ipcw_intercept"], r["ipcw_slope"] = ipcw_intercept_slope(T, d, C, p, c, h, G)
            r["e_over_o"] = expected_over_observed(p[fin], T[fin], d[fin], c, h)
            r["e_over_o_rows_left_out"] = int((~fin).sum())
            per_h[int(h)] = r
        blk = {"horizons": per_h}
        if ibs_too:
            blk["ibs"] = ibs({h: per_h[h]["brier_ipcw"] for h in per_h})
            blk["ibs_null_own"] = ibs({h: per_h[h]["null_own"]["bs_null"] for h in per_h})
            blk["ibs_ipa"] = ipa(blk["ibs"], blk["ibs_null_own"])
            blk["ibs_complete_case"] = ibs({h: per_h[h]["brier_complete_case"] for h in per_h})
        out["causes"][cname] = blk
    return out


def _train(pop: pd.DataFrame, g: str) -> pd.DataFrame:
    return pop[pop["group"] != g]


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _scorable(fr: dict, sset: str, cname: str, h: int, key: str, view: str = "primary") -> float | None:
    """The value of ``key`` in a JSON form of metrics_block (``fr[view][sset]``), or None when it is not scorable:
    no event of the cause by h, or (IPCW metrics and the weighted concordance) G fails the positivity rule at h."""
    blk = fr[view][sset]
    r = blk["causes"][cname]["horizons"][str(h)]
    if r["n_event_landmarks"] == 0:
        return None
    if key in IPCW_KEYS or (key.startswith("c_") and key != "c_unweighted"):
        if h in blk["G"]["not_scorable_horizons"]:
            return None
    if key.startswith("c_"):
        return r["c"][key[2:]]["c_index"]
    if key in ("ipa_training_null",):
        return r.get(key)
    return r[key]
