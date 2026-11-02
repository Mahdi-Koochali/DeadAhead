"""Tables of the supplement read from the predictions bundle, not from the reproduce output: the weight of each outcome
in the training rows, and two post hoc views of the warnings (the alarm share per 30-min bin before death, the match
group by group)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from deadahead.evaluation.aggregate import paired_both
from deadahead.evaluation.alarms import (CAUSE_IX, NOMINAL, WARNING_HORIZON, in_bin, sustained,
                                         threshold_and_apply)
from deadahead.evaluation.rows import CAUSES, GROUPS, LABEL_CENSORED, SEEDS
from deadahead.reproduce.alarm_rate import warning_inputs
from deadahead.reproduce.image_warnings import IMAGE, index_image_predictions
from deadahead.reproduce.warning_times import DEADAHEAD, index_predictions, warning_population

PRIMARY = "0.2_per_cell_hour"
BIN_MIN, BINS = 30, 8                         # eight 30-min bins before death, as in the warnings
LAST_HOUR = 60


def row_weight_table(bundle: dict) -> pd.DataFrame:
    """Per label: its share of the 30-min rows with a determined label, counted and under the training row weights."""
    r = bundle["rows"]
    d = r[(r["horizon_min"] == WARNING_HORIZON) & (r["label"] != LABEL_CENSORED)]
    w = d["window_weight"].to_numpy(float)
    rows = [{"label": lab, "share_of_rows_pct": 100 * float((d["label"] == lab).mean()),
             "share_of_weight_pct": 100 * float(w[(d["label"] == lab).to_numpy()].sum() / w.sum())}
            for lab in sorted(d["label"].unique())]
    return pd.DataFrame(rows)


def warning_input_rows(bundle: dict) -> dict:
    """The warning rows and, per method, group, seed and route, the reference and held-out scores."""
    index_predictions(bundle)
    index_image_predictions(bundle)
    rows, _ = warning_population(bundle)
    return warning_inputs(bundle, rows)


def alarm_share_table(inputs: dict) -> pd.DataFrame:
    """DeadAhead at the primary target: per route and 30-min bin before death, the share of the forecasts of dying
    cells with the alarm on, and the share of the held-out reference forecasts that raise an alarm (seed means, then
    means over the groups)."""
    out = []
    alpha = NOMINAL[PRIMARY]
    for c in CAUSES:
        share = {b: {} for b in range(1, BINS + 1)}
        reference = {}
        for g in GROUPS:
            rg = inputs["rows"][g]
            lead, cause = rg["lead_min"].to_numpy(float), rg["event_cause"].to_numpy()
            neg = rg["negative"].to_numpy(bool)
            per_seed = {b: [] for b in share}
            refs = []
            for s in SEEDS:
                ref, te = inputs["scores"][DEADAHEAD][(g, s, c)]
                flags, _ = threshold_and_apply(ref, te, alpha)
                refs.append(float(flags[neg].mean()))
                for b in share:
                    m = (cause == CAUSE_IX[c]) & in_bin(lead, BIN_MIN * (b - 1), BIN_MIN * b)
                    if m.any():
                        per_seed[b].append(float(flags[m].mean()))
            reference[g] = float(np.mean(refs))
            for b in share:
                if per_seed[b]:
                    share[b][g] = float(np.mean(per_seed[b]))
        ref_macro = float(np.mean(list(reference.values())))
        for b in share:
            out.append({"route": c, "bin_end_min": BIN_MIN * b, "n_groups": len(share[b]),
                        "alarm_share_pct": 100 * float(np.mean(list(share[b].values()))),
                        "reference_alarm_share_pct": 100 * ref_macro})
    return pd.DataFrame(out)


def group_match_table(inputs: dict, cause: str = "LYTIC") -> pd.DataFrame:
    """The match group by group: within each held-out group and seed, DeadAhead's threshold is read off that group's
    reference forecasts, so that it raises as many reference alarms as DINOv2 + handcrafted at the primary target. The
    share of deaths warned in the last hour per group (seed means), and one row with the macros and groups ahead."""
    alpha = NOMINAL[PRIMARY]
    dead, image = {}, {}
    for g in GROUPS:
        rg = inputs["rows"][g]
        neg = rg["negative"].to_numpy(bool)
        lead, ev = rg["lead_min"].to_numpy(float), rg["event_cause"].to_numpy()
        trk = rg["track_id"].astype(str).to_numpy()
        a, b = [], []
        for s in SEEDS:
            ref, te = inputs["scores"][IMAGE][(g, s, cause)]
            flags, _ = threshold_and_apply(ref, te, alpha)
            k = int(flags[neg].sum())                 # the baseline's alarms on this group's reference forecasts
            _, te_dead = inputs["scores"][DEADAHEAD][(g, s, cause)]
            ordered = np.sort(te_dead[neg])
            cut = ordered[max(ordered.size - k - 1, 0)]
            a.append(sustained(te_dead > cut, trk, lead, ev, CAUSE_IX[cause], alpha)[str(LAST_HOUR)]["share"])
            b.append(sustained(flags, trk, lead, ev, CAUSE_IX[cause], alpha)[str(LAST_HOUR)]["share"])
        dead[g], image[g] = float(np.mean(a)), float(np.mean(b))
    p = paired_both(dead, image)["nine_folds"]
    rows = [{"group": g.replace("_", " "), "deadahead_warned_pct": 100 * dead[g],
             "dinov2_handcrafted_warned_pct": 100 * image[g], "difference_points": 100 * p["per_group_difference"][g]}
            for g in GROUPS]
    rows.append({"group": "macro", "deadahead_warned_pct": 100 * p["macro_x"],
                 "dinov2_handcrafted_warned_pct": 100 * p["macro_y"], "difference_points": 100 * p["macro_difference"],
                 "groups_ahead": p["x_ahead"], "groups_behind": p["y_ahead"], "n_groups": p["n"]})
    return pd.DataFrame(rows).astype({"groups_ahead": "Int64", "groups_behind": "Int64", "n_groups": "Int64"})
