"""Outcome sets at a fixed horizon from the survival outcomes (T, delta, C in minutes), alarm counts on them, and
the per-unit and macro summaries of the fixed-horizon results."""

from __future__ import annotations

import numpy as np

from deadahead.evaluation.aggregate import LEFT_OUT_FOLD, both_folds
from deadahead.evaluation.alarms import CAUSE_IX, achieved_rate, pooled_rate
from deadahead.evaluation.rows import GROUPS, OTHER
from deadahead.evaluation.table1 import auprc_sets


def outcome_sets(T, delta, Cm, cause: str, h: float) -> dict:
    """P a death of the cause with T <= h, Q one of the other cause with T <= h, N no death by h and followed
    through h, U neither (follow-up ends inside h with no event); delta 1 apoptosis, 2 lytic, 0 none."""
    T, d, Cm = np.asarray(T, float), np.asarray(delta, int), np.asarray(Cm, float)
    if (d == 3).any():
        raise ValueError("an untyped death is in the evaluated rows: apply the cohort first")
    c, o = CAUSE_IX[cause], CAUSE_IX[OTHER[cause]]
    P = (d == c) & (T <= h)
    Q = (d == o) & (T <= h)
    N = ~(P | Q) & (Cm >= h)
    U = ~(P | Q | N)
    return {"P": P, "Q": Q, "N": N, "U": U}


def precision_sensitivity(flags, sets: dict) -> dict:
    """Alarm counts per set, precision over P, Q and N, sensitivity over P, and the prevalence |P| / |P + Q + N|."""
    f = np.asarray(flags, bool)
    aP, aQ, aN = int((f & sets["P"]).sum()), int((f & sets["Q"]).sum()), int((f & sets["N"]).sum())
    nP = int(sets["P"].sum())
    den = aP + aQ + aN
    return {"alarms_P": aP, "alarms_Q": aQ, "alarms_N": aN, "alarms_U_excluded": int((f & sets["U"]).sum()),
            "n_P": nP, "n_Q": int(sets["Q"].sum()), "n_N": int(sets["N"].sum()), "n_U": int(sets["U"].sum()),
            "precision": aP / den if den else float("nan"), "sensitivity": aP / nP if nP else float("nan"),
            "prevalence": nP / (nP + int(sets["Q"].sum()) + int(sets["N"].sum()))
            if (nP + int(sets["Q"].sum()) + int(sets["N"].sum())) else float("nan")}


def fixed_horizon_unit(flags, score, sets: dict) -> dict:
    """One (method, group, seed, cause, horizon, rate): precision, sensitivity, prevalence, AUPRC and the counts."""
    return {**precision_sensitivity(flags, sets), "auprc": auprc_sets(score, sets),
            "achieved": achieved_rate(flags, sets["N"])}


def fixed_horizon_summary(units: dict) -> dict:
    """{(g, s): fixed_horizon_unit} -> nine- and eight-group macros of precision, sensitivity, AUPRC, prevalence and
    the achieved alarm rate (equal-group and exposure-weighted); the groups with undefined precision (no alarm)
    or sensitivity (no positive) are named."""
    gs = sorted({g for g, _ in units})
    out = {}
    for k in ("precision", "sensitivity", "auprc", "prevalence"):
        pgs = {g: {s: u[k] for (gg, s), u in units.items() if gg == g} for g in gs}
        out[k] = both_folds(pgs)
    rate = {g: {s: u["achieved"]["rate_per_cell_hour"] for (gg, s), u in units.items() if gg == g} for g in gs}
    out["achieved_rate_equal_group_macro"] = both_folds(rate)
    counts = {g: {s: u["achieved"] for (gg, s), u in units.items() if gg == g} for g in gs}
    eight = tuple(g for g in GROUPS if g != LEFT_OUT_FOLD)
    out["achieved_rate_exposure_weighted"] = {"nine_folds": pooled_rate(counts, GROUPS),
                                              "eight_folds_without_PNT1A_bp": pooled_rate(counts, eight)}
    out["precision_undefined_groups"] = out["precision"]["nine_folds"]["groups_left_out"]
    out["sensitivity_undefined_groups"] = out["sensitivity"]["nine_folds"]["groups_left_out"]
    return out
