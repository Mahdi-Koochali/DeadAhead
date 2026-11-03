"""Reproduce the injected-censoring experiment (supplementary S6) from the bundle's stored unit predictions.

40 % administrative censoring is injected into the training landmarks of every held-out group; four regimes are
trained on the same censored data (handcrafted features) and scored against the uncensored oracle labels, 9 groups x
4 regimes x 3 seeds = 108 units. The survival regime's risk set ends at each landmark's known end of follow-up.
"""

from __future__ import annotations

import pandas as pd

from deadahead.evaluation.censoring import (CAUSES, REGIMES, macro_table, paired_contrasts, paper_side,
                                            unit_scores)
from deadahead.evaluation.metrics import auprc
from deadahead.evaluation.rows import GROUPS

LEVEL = {"mechanism": "administrative", "rate": 0.40}
CORRECTION_COUNTS = ("training_landmarks", "bins_at_risk_original", "bins_at_risk_corrected", "bins_removed",
                     "landmarks_truncated")
INJECTED_COUNTS = ("censor_bin_at_risk", "labelled_survived", "labelled_with_a_later_death_in_that_bin",
                   "event_rows")
PAPER_CONTRAST = "c_censoring_aware_minus_a_naive_drop"


def unit_name(g: str, regime: str, seed: int) -> str:
    """The display label of a unit in the output (the unit files are named ``<group>__<regime>__s<seed>``)."""
    return f"{g}|{regime}|s{int(seed)}"


def unit_payloads(preds: pd.DataFrame) -> dict:
    """{(group, seed, regime): {"y_<h>": labels, "p_<regime>_<cause>_<h>": scores}} from the long table, in the
    stored row order."""
    out: dict = {}
    for (g, s, r, h), d in preds.groupby(["group", "seed", "regime", "horizon_min"], sort=False):
        p = out.setdefault((g, int(s), r), {})
        p[f"y_{int(h)}"] = d["label"].to_numpy()
        for c in CAUSES:
            p[f"p_{r}_{c}_{int(h)}"] = d[f"score_{c}"].to_numpy(float)
    return out


def score_rows(bundle: dict) -> pd.DataFrame:
    """One AUPRC row per (unit, cause, horizon), the units in the order of the units table."""
    payloads = unit_payloads(bundle["censoring_predictions"])
    rows = []
    for u in bundle["censoring_units"].itertuples(index=False):
        key = (u.group, int(u.seed), u.regime)
        if key not in payloads:
            raise ValueError(f"the bundle holds no predictions of unit {unit_name(u.group, u.regime, u.seed)}")
        for r in unit_scores(payloads[key], u.regime, auprc):
            r.update(group=u.group, seed=int(u.seed))
            rows.append(r)
    return pd.DataFrame(rows)


def correction_summary(units: pd.DataFrame) -> dict:
    """The survival regime's risk-set rule over its 27 training sets: bins and landmarks removed by ending the risk
    set at the known end of follow-up, and how often an injected censor's own bin stays at risk."""
    d = units[units["regime"] == "d_joint_survival"]
    per_unit = {}
    for u in d.itertuples(index=False):
        per_unit[unit_name(u.group, u.regime, u.seed)] = {
            **{k: int(getattr(u, k)) for k in CORRECTION_COUNTS},
            "injected_censor_bin": {k: int(getattr(u, f"injected_{k}")) for k in INJECTED_COUNTS}}
    totals = {k: int(d[f"injected_{k}"].sum()) for k in INJECTED_COUNTS}
    return {"training_bins_removed_total": int(d["bins_removed"].sum()),
            "training_landmarks_truncated_total": int(d["landmarks_truncated"].sum()),
            "per_unit": per_unit,
            "injected_censor_bin": {
                "totals": totals,
                "labelled_with_a_later_death_in_that_bin_share_of_event_rows": (
                    None if not totals["event_rows"] else
                    totals["labelled_with_a_later_death_in_that_bin"] / totals["event_rows"])}}


def part_censoring(bundle: dict) -> dict:
    """The injected-censoring experiment from the bundle's stored unit predictions."""
    if "censoring_predictions" not in bundle or "censoring_units" not in bundle:
        raise ValueError("the bundle holds no injected-censoring predictions: the experiment cannot be scored")
    units = bundle["censoring_units"]
    rows = score_rows(bundle)
    groups = [g for g in GROUPS if g in set(units["group"])]
    table = macro_table(rows)
    contrasts = paired_contrasts(rows)
    return {"level": LEVEL, "regimes": list(REGIMES), "seeds": sorted(int(s) for s in units["seed"].unique()),
            "groups": groups, "units": int(len(units)),
            "metric": "oracle AUPRC per (group, seed); the mean over seeds within a group, then over the groups",
            "results_by_regime": table, "contrasts": contrasts,
            "paper_paragraph": {
                "what": "c_censoring_aware minus a_naive_drop, 30-min oracle AUPRC",
                "values": {c: paper_side(contrasts.get(f"{PAPER_CONTRAST}|{c}|h30"), len(groups)) for c in CAUSES}},
            "achieved_censoring_rate": {unit_name(u.group, u.regime, u.seed): float(u.achieved_censoring_rate)
                                        for u in units.itertuples(index=False)},
            "correction": correction_summary(units)}


def format_censoring(res: dict) -> str:
    """The paired contrasts at 30 and 240 min as text."""
    out = [f"Injected censoring ({int(100 * res['level']['rate'])} % {res['level']['mechanism']}, {res['units']} "
           f"units): paired contrasts of oracle AUPRC, mean over {len(res['groups'])} groups (groups ahead)"]
    for key in ("c_censoring_aware_minus_a_naive_drop", "c_censoring_aware_minus_b_naive_mislabel",
                "d_joint_survival_minus_c_censoring_aware"):
        cells = []
        for c in ("LYTIC", "APOPTOSIS"):
            for h in (30, 240):
                v = res["contrasts"].get(f"{key}|{c}|h{h}")
                cells.append(f"{c.lower()} {h} min {v['mean_diff']:+.3f} ({v['wins']}/{v['n_units']})" if v else
                             f"{c.lower()} {h} min -")
        out.append(f"  {key}: " + ", ".join(cells))
    out.append("a_naive_drop: deleting censored forecasts; b_naive_mislabel: counting censored forecasts as negatives; "
               "c_censoring_aware: censoring-aware training; d_joint_survival: the survival head, not reported (S6d)")
    return "\n".join(out)
