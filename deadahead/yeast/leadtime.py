"""Lead time of a death forecast on one held-out position.

The threshold is the (1 - a) percentile of a score on negative windows (landmarks observed event-free for the full
240 min after t), a = 0.05 (0.01 and 0.10 beside). The alarm onset is the first landmark above the threshold from
which the score stays above it until the event, with at most one landmark (30 min) at or below it after the onset.
Lead time = event time - alarm onset, capped at 16 h; a death never flagged counts as 0. Also: the false-alarm rate
reached, alarms per 100 lifespan-hours of negative time, the share of event-free lifespans that ever alarm, the buds
between onset and death, and a time-shuffle check (each lifespan's scores shuffled, the threshold unchanged).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C

A_MAIN = 0.05
A_LEVELS = (0.05, 0.01, 0.10)
L_MAX_MIN = 16 * 60
FLAG_BY = (30, 60, 120, 240)
ALLOW_BELOW = 1
H_STAR = 240


def threshold(neg_scores, a: float = A_MAIN) -> float:
    v = np.asarray(neg_scores, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        raise ValueError("no negative window to set a threshold on")
    return float(np.percentile(v, 100.0 * (1.0 - a)))


def alarm_onset(scores, thr: float, allow_below: int = ALLOW_BELOW) -> int | None:
    """Index of the first landmark above the threshold from which at most ``allow_below`` later landmarks are at or
    below it (up to the last landmark before the event); None when the death is never flagged."""
    above = np.asarray(scores, dtype=float) > thr
    n = len(above)
    below_after = np.zeros(n, dtype=int)                 # landmarks at or below the threshold after index i
    for i in range(n - 2, -1, -1):
        below_after[i] = below_after[i + 1] + int(not above[i + 1])
    ok = np.where(above & (below_after <= allow_below))[0]
    return int(ok[0]) if len(ok) else None


def lead_times(lifespans: list[dict], thr: float) -> dict:
    """``lifespans``: dicts with t (landmark frames, ascending), s (scores), event_frame (or None), buds (bud
    frames) and neg (bool per landmark: a negative window). Returns the lead-time table of one model on one position."""
    leads, buds_between, never, at_cap = [], [], 0, 0
    neg_scores, n_alarm, n_neg = [], 0, 0
    free, free_alarm = 0, 0
    for ls in lifespans:
        t, s = np.asarray(ls["t"]), np.asarray(ls["s"], dtype=float)
        neg = np.asarray(ls["neg"], dtype=bool)
        above = s > thr
        neg_scores.extend(s[neg].tolist())
        n_neg += int(neg.sum())
        starts = above & np.r_[True, ~above[:-1]]          # an alarm episode starts here
        n_alarm += int((starts & neg).sum())
        if ls["event_frame"] is None:
            free += 1
            free_alarm += int(above.any())
            continue
        i = alarm_onset(s, thr)
        if i is None:
            never += 1
            leads.append(0.0)
            continue
        lead = float((ls["event_frame"] - t[i]) * C.FRAME_MIN)
        at_cap += int(lead >= L_MAX_MIN)
        leads.append(min(lead, L_MAX_MIN))
        b = np.asarray(ls.get("buds", []))
        buds_between.append(int(((b > t[i]) & (b <= ls["event_frame"])).sum()))
    leads_a = np.asarray(leads)
    neg_a = np.asarray(neg_scores)
    return {"n_deaths": int(len(leads)), "n_never_flagged": int(never),
            "median_lead_min": float(np.median(leads_a)) if len(leads_a) else None,
            "share_at_cap": float(at_cap / len(leads)) if leads else None,
            "share_flagged_by": {str(h): float((leads_a >= h).mean()) if len(leads_a) else None for h in FLAG_BY},
            "false_alarm_rate": float((neg_a > thr).mean()) if len(neg_a) else None,
            "alarms_per_100_negative_hours": float(100.0 * n_alarm / (0.5 * n_neg)) if n_neg else None,
            "share_eventfree_lifespans_ever_alarm": float(free_alarm / free) if free else None,
            "median_buds_onset_to_death": float(np.median(buds_between)) if buds_between else None}


def shuffled(lifespans: list[dict], seed: int = C.SEED) -> list[dict]:
    """Each lifespan's score sequence shuffled in time (one seeded generator over the lifespans in order)."""
    rng = np.random.default_rng(seed)
    return [{**ls, "s": rng.permutation(np.asarray(ls["s"], dtype=float))} for ls in lifespans]


def lifespans_of(labels: pd.DataFrame, scores: np.ndarray, events: dict, buds: dict) -> list[dict]:
    """Group one position's rows (unit, t, y_240; in any order) with a score per row into lifespans."""
    d = labels[["unit", "t", f"y_{H_STAR}"]].assign(s=np.asarray(scores, dtype=float)).sort_values(["unit", "t"])
    out = []
    for u, g in d.groupby("unit", sort=True):
        out.append({"unit": u, "t": g["t"].to_numpy(), "s": g["s"].to_numpy(), "event_frame": events.get(u),
                    "buds": buds.get(u, np.zeros(0, int)), "neg": (g[f"y_{H_STAR}"] == 0).to_numpy()})
    return out


def table(lifespans: list[dict], thresholds: dict) -> dict:
    """The lead-time table at every a level, and the time-shuffle check at the main level."""
    out = {f"a={a:g}": {"threshold": thresholds[a], **lead_times(lifespans, thresholds[a])} for a in A_LEVELS}
    out["time_shuffle_a=0.05"] = lead_times(shuffled(lifespans), thresholds[A_MAIN])
    return out


def thresholds_from(neg_scores) -> dict:
    return {a: threshold(neg_scores, a) for a in A_LEVELS}


def median_lead_bootstrap(per_model_lifespans: list[list[dict]], thresholds: list[float], *, n_boot: int = 2000,
                          seed: int = C.SEED) -> list[float] | None:
    """A 95% interval of the row's median lead time on one position (the mean over its models of each model's
    median lead), resampling whole lifespans (the same draw for every model)."""
    if not per_model_lifespans or not per_model_lifespans[0]:
        return None
    rng = np.random.default_rng(seed)
    n = len(per_model_lifespans[0])
    per_model = []
    for ls, thr in zip(per_model_lifespans, thresholds):
        lead = []
        for x in ls:
            if x["event_frame"] is None:
                lead.append(np.nan)
                continue
            i = alarm_onset(x["s"], thr)
            lead.append(0.0 if i is None else min(float((x["event_frame"] - x["t"][i]) * C.FRAME_MIN), L_MAX_MIN))
        per_model.append(np.asarray(lead))
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        vals = [np.nanmedian(v[pick]) for v in per_model if np.isfinite(v[pick]).any()]
        if vals:
            draws.append(float(np.mean(vals)))
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if draws else None
