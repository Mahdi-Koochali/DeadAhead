"""Reproduce the sensitivity analyses that the paper quotes: Table 1 restricted to rows whose 30-min window ends
inside the common cut (frame 438), and the warnings at a common landmark (rows t <= 358, follow-up truncated at
frame 438, thresholds reselected on the restricted training reference rows).
"""

from __future__ import annotations

import numpy as np

from deadahead.evaluation.aggregate import outcome_text
from deadahead.evaluation.alarms import (CAUSE_IX, CUT_FRAME, NOMINAL, PRIMARY_RATE, WARNING_HORIZON, sustained,
                                         warning_rows)
from deadahead.evaluation.rows import CAUSES, GROUPS, LABEL_CENSORED, TABLE1_HORIZON
from deadahead.reproduce.horizons import format_horizons
from deadahead.reproduce.table1 import METHOD_ORDER, part_table1
from deadahead.reproduce.warning_times import SUSTAINED_CSV_X, index_predictions, rate_label, warnings_block

COMMON_CUT = "4.1_fixed_horizon_common_cut"     # the fixed-horizon section (horizon inputs)


def population_change(main_rows, rows) -> dict:
    """The change from the main warning rows to the common landmark's, per cause: reference rows and the deaths
    eligible at each X (a forecast row in every 30-min bin up to X), per group."""
    out = {"reference_rows": {"main": int(main_rows["negative"].sum()), "common_landmark": int(rows["negative"].sum())},
           "eligible_deaths": {}}
    for c in CAUSES:
        for which, r in (("main", main_rows), ("common_landmark", rows)):
            for g in GROUPS:
                rg = r[r["group"] == g]
                sus = sustained(np_zeros(len(rg)), rg["track_id"].astype(str).to_numpy(), rg["lead_min"].to_numpy(),
                                rg["event_cause"].to_numpy(), CAUSE_IX[c], NOMINAL[PRIMARY_RATE])
                for X, v in sus.items():
                    out["eligible_deaths"].setdefault(c, {}).setdefault(X, {}).setdefault(which, {})[g] = \
                        v["n_eligible_events"]
    return out


def np_zeros(n: int):
    return np.zeros(n, bool)


def cut30(rows):
    """The rows whose 30-min window ends inside the common cut, t + 10 <= 438."""
    return rows[rows["t_obs_frame"] + TABLE1_HORIZON / 3.0 <= CUT_FRAME].reset_index(drop=True)


def part_sensitivity(bundle: dict) -> dict:
    """Table 1 on the rows with t + 10 <= 438, and the warnings at the common landmark with the population change."""
    if "inner_predictions" not in bundle or "movies" not in bundle:
        raise ValueError("the bundle holds no inner predictions or movie table: the sensitivity cannot be computed")
    out = {}
    t1 = part_table1(bundle, rows_filter=cut30)
    out["4.1_table1_common_cut_30min"] = {"rows": "Table 1's rows with t + 10 <= 438 (labels already at the cut)",
                                          **t1}
    index_predictions(bundle)
    r = bundle["rows"]
    determined = r[(r["horizon_min"] == WARNING_HORIZON) & (r["label"] != LABEL_CENSORED)].reset_index(drop=True)
    main_rows, _ = warning_rows(determined, bundle["events"], bundle["movies"], bundle["untyped"])
    rows, info = warning_rows(determined, bundle["events"], bundle["movies"], bundle["untyped"], common_landmark=True)
    blk = warnings_block(bundle, rows, info, time_base=main_rows)
    blk["population_change"] = population_change(main_rows, rows)
    out["4.2_warnings_common_landmark"] = {
        "rows": "t <= 358, labels truncated at 438, roles and eligible deaths recomputed, thresholds reselected on "
                "the restricted training reference rows (the same cross-fitted inner scores, masked)", **blk}
    return out


def format_sensitivity(res: dict) -> str:
    """The sensitivity analyses as text: the AUPRC of each method and DeadAhead minus each method in Table 1 at the
    common cut, the warnings at the common landmark with DeadAhead minus elapsed time, and the fixed-horizon results
    at the common cut."""
    t1 = res["4.1_table1_common_cut_30min"]["cohort"]
    lines = [f"Sensitivity: Table 1 on the rows with t + 10 <= 438 ({t1['n_rows']} rows)",
             f"{'method':<42}{'lytic':>8}{'apoptosis':>11}"]
    for m in METHOD_ORDER:
        if m in t1["methods"]:
            b = t1["methods"][m]
            lines.append(f"{m:<42}{b['LYTIC']['nine_folds']['macro']:>8.3f}"
                         f"{b['APOPTOSIS']['nine_folds']['macro']:>11.3f}")
    lines += ["", "DeadAhead minus each method: macro difference, groups ahead of 9, outcome"]
    for k, v in t1["contrasts"].items():
        if not k.startswith("deadahead - "):
            continue
        p = v["nine_folds"]
        lines.append(f"  {k:<52}{p.get('macro_difference', float('nan')):+8.3f}{p.get('x_ahead', 0):>4d}/{p['n']}  "
                     f"{outcome_text(p)}")
    w = res["4.2_warnings_common_landmark"]
    lines += ["", (f"Sensitivity: warnings at the common landmark (t <= 358; {w['rows']['n']} rows, "
                   f"{w['rows']['reference_rows']} reference rows)")]
    for rate in NOMINAL:
        for c in ("LYTIC", "APOPTOSIS"):
            for m in ("deadahead", "elapsed_time"):
                b = w["models"][m][c][rate]
                ach = b["achieved_rate_exposure_weighted"]["nine_folds"]["rate_per_cell_hour"]
                cov = "  ".join(f"{X}:{100 * b['sustained'][X]['nine_folds']['macro']:5.1f}" for X in b["sustained"])
                lines.append(f"  {rate_label(rate):<12}{c.lower():<11}{m:<14} achieved {ach:.2f} pooled  {cov}")
    lines += ["", "DeadAhead minus elapsed time at the common landmark: macro difference (percentage points), groups "
                  "ahead of 9"]
    for c in ("LYTIC", "APOPTOSIS"):
        cells = []
        for X in SUSTAINED_CSV_X:
            p = w["contrasts"][f"deadahead - elapsed_time|{c}|{PRIMARY_RATE}|sustained {X}"]["nine_folds"]
            cells.append(f"{X}:{100 * p.get('macro_difference', float('nan')):+6.1f} {p.get('x_ahead', 0)}/{p['n']}")
        lines.append(f"  {rate_label(PRIMARY_RATE):<12}{c.lower():<11}" + "  ".join(cells))
    if COMMON_CUT in res:
        lines += ["", "Sensitivity: the fixed-horizon results at the common cut (rows t + h/3 <= 438, labels truncated "
                      "at 438, thresholds of the main analysis)", format_horizons(res[COMMON_CUT])]
    return "\n".join(lines)
