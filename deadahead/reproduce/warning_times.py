"""Reproduce the warnings from a predictions bundle: training-only alarm thresholds at targets of 0.2 and 0.4 alarms per
event-free reference cell-hour, applied to DeadAhead's held-out F_c(30) and to the elapsed-time score.

DeadAhead's thresholds come from its cross-fitted inner predictions on the training groups' reference rows (each
outer group and seed has eight inner heads, one per training group); the elapsed-time score uses its own
thresholds on the same reference rows.
"""

from __future__ import annotations

import numpy as np

from deadahead.evaluation.aggregate import paired_both
from deadahead.evaluation.alarms import (NOMINAL, WARNING_HORIZON, reference_record, threshold_and_apply,
                                         time_scores, warning_rows, warning_summary, warning_unit)
from deadahead.evaluation.rows import CAUSES, GROUPS, LABEL_CENSORED, SEEDS, align_scores

DEADAHEAD, TIME = "deadahead", "elapsed_time"
TIME_SEED = 0
SUSTAINED_CSV_X = ("30", "60", "90", "120", "180", "240")


def rate_label(rate: str) -> str:
    """A nominal rate as printed: '0.2_per_cell_hour' -> 'target 0.2'."""
    return f"target {rate.split('_', 1)[0]}"


def inner_ref_scores(bundle: dict, g: str, s: int, rows, ref_mask, cause: str):
    """The pooled cross-fitted inner scores of outer (g, s) on the training groups' reference rows."""
    parts = []
    m = np.asarray(ref_mask, bool)
    inner = bundle["inner_by_unit"]
    for h in [x for x in GROUPS if x != g]:
        rh = rows[(rows["group"] == h).to_numpy() & m]
        z = inner[(g, s, h, cause)]
        parts.append(align_scores(z, z["score"].to_numpy(float), rh))
    return np.concatenate(parts) if parts else np.zeros(0)


def time_for(rows, base=None):
    """The elapsed-time score of ``rows``: one draw over ``base`` when given, so a row keeps its score in a
    restricted population, else one draw over ``rows``."""
    if base is None:
        return time_scores(rows).to_numpy()
    return align_scores(base, time_scores(base).to_numpy(), rows)


def outer_scores(bundle: dict, g: str, s: int, cause: str, rg):
    """DeadAhead's held-out F_c(30) of outer (g, s) at ``rg``'s keys."""
    z = bundle["outer_by_unit"][(g, s, cause)]
    return align_scores(z, z["score"].to_numpy(float), rg)


def warnings_block(bundle: dict, rows, info: dict, time_base=None) -> dict:
    """Thresholds, alarms and coverage for DeadAhead and elapsed time over the warning rows."""
    neg = rows["negative"].to_numpy(bool)
    ts = time_for(rows, time_base)
    out = {"rows": {"n": len(rows), "reference_rows": int(neg.sum()),
                    "warning_window": int(rows["warning_window"].sum()),
                    "censoring_bucket": int(rows["censoring_bucket"].sum()), **info},
           "reference_populations": {}, "thresholds": {}, "models": {}}
    units = {}
    for g in GROUPS:
        out["reference_populations"][g] = reference_record(rows, neg, g, info.get("cohort_removed"))
        gm = (rows["group"] == g).to_numpy()
        rg = rows[gm].reset_index(drop=True)
        for c in CAUSES:
            for rate, a in NOMINAL.items():
                for s in SEEDS:
                    ref = inner_ref_scores(bundle, g, s, rows, neg, c)
                    te = outer_scores(bundle, g, s, c, rg)
                    flags, thr = threshold_and_apply(ref, te, a)
                    out["thresholds"][f"{DEADAHEAD}|{g}|s{s}|{c}|{rate}"] = thr
                    units.setdefault((DEADAHEAD, c, rate), {})[(g, s)] = warning_unit(flags, rg, c, a)
                flags, thr = threshold_and_apply(ts[(~gm) & neg], ts[gm], a)
                out["thresholds"][f"{TIME}|{g}|{c}|{rate}"] = thr
                units.setdefault((TIME, c, rate), {})[(g, TIME_SEED)] = warning_unit(flags, rg, c, a)
    for (m, c, rate), u in units.items():
        out["models"].setdefault(m, {}).setdefault(c, {})[rate] = {
            **warning_summary(u), "units": {f"{g}|s{s}": v for (g, s), v in u.items()}}
    out["contrasts"] = {}
    for c in CAUSES:
        for rate in NOMINAL:
            b, t = out["models"][DEADAHEAD][c][rate], out["models"][TIME][c][rate]
            for X in b["sustained"]:
                out["contrasts"][f"{DEADAHEAD} - {TIME}|{c}|{rate}|sustained {X}"] = paired_both(
                    b["sustained"][X]["nine_folds"]["per_group"], t["sustained"][X]["nine_folds"]["per_group"])
    return out


def index_predictions(bundle: dict) -> None:
    """Split the outer and inner prediction tables once by unit, for the per-unit lookups above."""
    if "outer_by_unit" not in bundle:
        p = bundle["predictions"][DEADAHEAD]
        p = p[p["cause"].isin(CAUSES)]
        bundle["outer_by_unit"] = {(g, int(s), c): d for (g, s, c), d in p.groupby(["group", "seed", "cause"])}
    if "inner_by_unit" not in bundle:
        q = bundle["inner_predictions"]
        bundle["inner_by_unit"] = {(g, int(s), h, c): d for (g, s, h, c), d in
                                   q.groupby(["outer_group", "seed", "inner_group", "cause"])}


def warning_population(bundle: dict) -> tuple:
    """The warning rows (the 30-min rows with a determined label, on the common evaluation cohort) and their counts."""
    r = bundle["rows"]
    determined = r[(r["horizon_min"] == WARNING_HORIZON) & (r["label"] != LABEL_CENSORED)].reset_index(drop=True)
    return warning_rows(determined, bundle["events"], bundle["movies"], bundle["untyped"])


def part_warnings(bundle: dict) -> dict:
    """The warnings over the 30-min rows with a determined label, on the common evaluation cohort; reference rows:
    no death of either type in the next 240 min and >= 240 min of complete-movie follow-up."""
    if "inner_predictions" not in bundle or "movies" not in bundle:
        raise ValueError("the bundle holds no inner predictions or movie table: the warnings cannot be computed")
    index_predictions(bundle)
    rows, info = warning_population(bundle)
    return {"population": "the 30-min rows with a determined label, common evaluation cohort; reference rows: no "
                          "death of either type in the next 240 min and >= 240 min of complete-movie follow-up",
            **warnings_block(bundle, rows, info)}


def format_warnings(res: dict) -> str:
    """The warnings as text: per target and cause the achieved rates, the deaths warned X min ahead of each method, the
    same without PNT1A_bp, the eligible deaths and DeadAhead minus elapsed time."""
    lines = ["Warnings: deaths warned X min ahead (%, mean over the 9 groups), training-only thresholds; targets and "
             "achieved rates in alarms per event-free reference cell-hour", ""]
    for rate in NOMINAL:
        for c in ("LYTIC", "APOPTOSIS"):
            lines.append(f"{rate_label(rate)}, {c.lower()}")
            for m in (DEADAHEAD, TIME):
                b = res["models"][m][c][rate]
                ach = b["achieved_rate_exposure_weighted"]["nine_folds"]["rate_per_cell_hour"]
                mean = b["achieved_rate_equal_group_macro"]["nine_folds"]["macro"]
                cov = "  ".join(f"{X}:{100 * b['sustained'][X]['nine_folds']['macro']:5.1f}" for X in SUSTAINED_CSV_X)
                eight = "  ".join(f"{X}:{100 * b['sustained'][X]['eight_folds_without_PNT1A_bp']['macro']:5.1f}"
                                  for X in SUSTAINED_CSV_X)
                lines.append(f"  {m:<14} achieved {ach:.2f} pooled, {mean:.2f} mean   {cov}")
                lines.append(f"  {'':<14} {'without PNT1A_bp':<34}{eight}")
            sus = res["models"][DEADAHEAD][c][rate]["sustained"]
            elig = [f"{X}:{sum(sus[X]['eligible_deaths_per_group'].values()):5d}" for X in SUSTAINED_CSV_X]
            lines.append(f"  {'eligible deaths':<49}" + "  ".join(elig))
            diff = []
            for X in SUSTAINED_CSV_X:
                p = res["contrasts"][f"{DEADAHEAD} - {TIME}|{c}|{rate}|sustained {X}"]["nine_folds"]
                diff.append(f"{X}:{100 * p.get('macro_difference', float('nan')):+5.1f} {p.get('x_ahead', 0)}/{p['n']}")
            lines.append(f"  {'DeadAhead minus elapsed time':<49}" + "  ".join(diff))
    return "\n".join(lines)
