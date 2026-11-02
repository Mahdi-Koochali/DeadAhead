"""Table 1: AUPRC at 30 min per method, cause, held-out group and seed, the macros and the paired contrasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from deadahead.evaluation.aggregate import both_folds, paired_both
from deadahead.evaluation.metrics import auprc
from deadahead.evaluation.rows import CAUSES, set_counts, window_sets


def auprc_sets(score, sets: dict) -> float:
    """AUPRC over P, Q and N (U excluded; Q and N are the negatives); nan with no positive."""
    s = np.asarray(score, float)
    m = sets["P"] | sets["Q"] | sets["N"]
    if not sets["P"].any():
        return float("nan")
    return float(auprc(sets["P"][m].astype(int), s[m]))


def table1_unit(score, label, cause: str) -> dict:
    """One (method, group, seed, cause) on one row set: AUPRC over P, Q, N and the set counts."""
    sets = window_sets(label, cause)
    return {"auprc": auprc_sets(score, sets), **set_counts(sets)}


def table1(scores: dict, rows: pd.DataFrame, method_order=None) -> dict:
    """``scores``: {method: {(g, s): {cause: array aligned to rows[rows.group == g]}}}; ``rows`` the evaluated rows
    (KEY, group, label). Per method and cause: per (g, s) AUPRC and counts, the nine- and eight-group macros; the
    paired contrasts of the first two methods of ``method_order`` against every other method."""
    out = {"methods": {}, "contrasts": {}}
    for m, by_gs in scores.items():
        blk = {}
        for cause in CAUSES:
            pgs, units = {}, {}
            for (g, s), sc in sorted(by_gs.items()):
                lab = rows.loc[rows["group"] == g, "label"].to_numpy()
                u = table1_unit(sc[cause], lab, cause)
                units[f"{g}|s{s}"] = u
                pgs.setdefault(g, {})[s] = u["auprc"]
            blk[cause] = {"units": units, **both_folds(pgs)}
        out["methods"][m] = blk
    names = list(scores) if method_order is None else [m for m in method_order if m in scores]
    for x in names[:2]:
        for y in names:
            if y == x:
                continue
            for cause in CAUSES:
                px = out["methods"][x][cause]["nine_folds"]["per_group"]
                py = out["methods"][y][cause]["nine_folds"]["per_group"]
                out["contrasts"][f"{x} - {y}|{cause}"] = paired_both(px, py)
    return out
