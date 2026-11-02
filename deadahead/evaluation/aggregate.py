"""Aggregation over held-out groups: seed means, the macro over groups, paired differences and the decision rule.

The unit is the held-out group (n = 9). A value is averaged over seeds within its group, then over groups. Two
methods are compared by their per-group differences: the macro difference, the number of groups where each is
ahead, and the decision rule (|macro difference| >= 0.02 with the same sign in >= 7 groups).
"""

from __future__ import annotations

import math

import numpy as np

from deadahead.evaluation.rows import GROUPS

LEFT_OUT_FOLD = "PNT1A_bp"                   # the eight-fold print leaves this group out
MARGIN = 0.02
COUNT = 7                                    # groups with the same sign, of 9
MIN_GROUPS = 7                               # fewer groups on both sides: not scorable

#: outcome labels of a paired comparison (the paper's wording)
AHEAD_X = "x ahead (meets the decision rule)"
AHEAD_Y = "y ahead (meets the decision rule)"
NOT_IMPORTANT = "below the decision rule"
NOT_SCORABLE = "NOT SCORABLE"


def _finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def macro(per_group_seed: dict, groups=GROUPS) -> dict:
    """{g: {s: v}} -> per-group seed means, the macro over the groups given, the groups left out (named, with why)."""
    per_group, dropped = {}, {}
    for g in groups:
        if g not in per_group_seed or not per_group_seed[g]:
            dropped[g] = "no value"
            continue
        vals = list(per_group_seed[g].values())
        if not all(_finite(v) for v in vals):
            dropped[g] = f"undefined at {sum(not _finite(v) for v in vals)} of {len(vals)} seed(s)"
            continue
        per_group[g] = float(np.mean([float(v) for v in vals]))
    keep = [g for g in groups if g in per_group]
    return {"macro": float(np.mean([per_group[g] for g in keep])) if keep else float("nan"),
            "per_group": per_group, "n_groups": len(keep), "groups_contributing": keep,
            "groups_left_out": dropped, "n_left_out": len(dropped)}


def both_folds(per_group_seed: dict) -> dict:
    """The nine-group macro and, beside it, the macro over the eight groups other than PNT1A_bp."""
    return {"nine_folds": macro(per_group_seed, GROUPS),
            "eight_folds_without_PNT1A_bp": macro(per_group_seed, tuple(g for g in GROUPS if g != LEFT_OUT_FOLD))}


def paired(x: dict, y: dict, groups=GROUPS, count: int = COUNT, min_groups: int = MIN_GROUPS) -> dict:
    """x - y over the groups both contribute (per-group values, already seed means): the per-group differences,
    the macro difference, groups ahead and the decision-rule outcome."""
    gs = [g for g in groups if g in x and g in y and _finite(x[g]) and _finite(y[g])]
    d = np.array([float(x[g]) - float(y[g]) for g in gs], float)
    n = len(gs)
    out = {"n": n, "groups": gs, "groups_left_out": [g for g in groups if g not in gs],
           "per_group_difference": dict(zip(gs, d.tolist()))}
    if n == 0:
        return {**out, "outcome": NOT_SCORABLE, "reason": "no group on both sides"}
    delta = float(d.mean())
    kp, km = int((d > 0).sum()), int((d < 0).sum())
    out.update(macro_difference=delta, x_ahead=kp, y_ahead=km, ties=n - kp - km,
               macro_x=float(np.mean([x[g] for g in gs])), macro_y=float(np.mean([y[g] for g in gs])),
               rule=f"|macro difference| >= {MARGIN} and the same sign in >= {count} groups")
    if n < min_groups:
        out.update(outcome=NOT_SCORABLE, reason=f"n = {n} < {min_groups}")
    elif delta >= MARGIN and kp >= count:
        out["outcome"] = AHEAD_X
    elif delta <= -MARGIN and km >= count:
        out["outcome"] = AHEAD_Y
    else:
        out["outcome"] = NOT_IMPORTANT
    return out


def outcome_text(p: dict, x: str = "DeadAhead") -> str:
    """The outcome of a paired comparison for printing, with the x side named."""
    return p["outcome"].replace("x ahead", f"{x} ahead").replace("y ahead", "the other method ahead")


def paired_both(x: dict, y: dict) -> dict:
    """The nine-group paired comparison and the same arithmetic over the eight groups other than PNT1A_bp."""
    eight = tuple(g for g in GROUPS if g != LEFT_OUT_FOLD)
    return {"nine_folds": paired(x, y, GROUPS),
            "eight_folds_without_PNT1A_bp": paired(x, y, eight)}
