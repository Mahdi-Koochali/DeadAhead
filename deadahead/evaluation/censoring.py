"""Scores of the injected-censoring experiment: oracle AUPRC per unit, the macro over groups and paired contrasts.

A unit is one (held-out group, regime, seed). Its scores are compared with the oracle labels read from the complete
movies (0 no event, 1 apoptosis, 2 lytic death). A value is averaged over seeds within its group, then over groups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CAUSES = ("APOPTOSIS", "LYTIC")
CAUSE_IX = {"APOPTOSIS": 1, "LYTIC": 2}
#: the four training regimes: (a) drop every censored forecast, (b) label censored windows event-free, (c) drop only the
#: horizons whose label is unknown, (d) the discrete-time survival head with censoring in its risk set
REGIMES = ("a_naive_drop", "b_naive_mislabel", "c_censoring_aware", "d_joint_survival")
#: the paired contrasts, each r1 minus r2
CONTRASTS = (("c_censoring_aware", "a_naive_drop"), ("c_censoring_aware", "b_naive_mislabel"),
             ("d_joint_survival", "c_censoring_aware"))
HORIZONS = (30, 60, 120, 240)


def unit_scores(payload: dict, regime: str, auprc_fn, hs=(30, 60, 120, 240)) -> list:
    """AUPRC rows of one unit and one regime: y_h against p_<regime>_<cause>_h, the survival regime aligned to y_h
    by its d_mask_h; a (cause, h) with no positive is left out."""
    out = []
    for h in hs:
        if f"y_{h}" not in payload:
            continue
        y = np.asarray(payload[f"y_{h}"])
        for c in CAUSES:
            k = f"p_{regime}_{c}_{h}"
            if k not in payload:
                continue
            p = np.asarray(payload[k])
            if regime == "d_joint_survival" and f"d_mask_{h}" in payload and len(p) != len(y):
                p = p[np.asarray(payload[f"d_mask_{h}"], bool)]
            if len(p) != len(y):
                continue
            yt = (y == CAUSE_IX[c]).astype(float)
            if yt.sum() == 0:
                continue
            out.append({"regime": regime, "cause": c, "horizon_min": int(h), "auprc": float(auprc_fn(yt, p)),
                        "n": int(len(yt)), "n_pos": int(yt.sum())})
    return out


def macro_table(rows: pd.DataFrame) -> dict:
    """{regime: {cause: {h: {macro, per_group, n_groups}}}}: the mean over seeds within a group, then over groups."""
    out: dict = {}
    by_g = rows.groupby(["regime", "cause", "horizon_min", "group"])["auprc"].mean().reset_index()
    for (r, c, h), d in by_g.groupby(["regime", "cause", "horizon_min"]):
        out.setdefault(r, {}).setdefault(c, {})[str(int(h))] = {
            "macro": float(d["auprc"].mean()), "n_groups": int(len(d)),
            "per_group": {str(g): float(v) for g, v in zip(d["group"], d["auprc"])}}
    return out


def paired_contrasts(rows: pd.DataFrame) -> dict:
    """r1 - r2 per (cause, h), paired over groups: the mean difference, the groups where r1 is ahead, the number of
    groups and the per-group differences."""
    out: dict = {}
    for (c, h), sel in rows.groupby(["cause", "horizon_min"]):
        g = sel.groupby(["regime", "group"])["auprc"].mean().unstack(0)
        for r1, r2 in CONTRASTS:
            if r1 not in g or r2 not in g:
                continue
            d = (g[r1] - g[r2]).dropna()
            out[f"{r1}_minus_{r2}|{c}|h{int(h)}"] = {
                "mean_diff": float(d.mean()), "wins": int((d > 0).sum()), "n_units": int(len(d)),
                "per_group": {str(k): float(v) for k, v in d.items()}}
    return out


def paper_side(v: dict | None, n_groups: int) -> dict | None:
    """One contrast as the paper reports it: the mean and the groups ahead, only when all ``n_groups`` groups
    contribute."""
    if not v:
        return None
    if v["n_units"] != n_groups:
        return {"mean_diff": None, "groups_ahead": None, "of": v["n_units"],
                "withheld": f"fewer than {n_groups} groups contribute"}
    return {"mean_diff": v["mean_diff"], "groups_ahead": v["wins"], "of": v["n_units"]}
