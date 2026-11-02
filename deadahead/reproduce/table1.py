"""Reproduce Table 1 from a predictions bundle: AUPRC at 30 min on the common evaluation cohort, every method.

The image comparators enter twice: ``<method>|refit`` (the Table 1 entry: the prediction head refit without the
untyped-death training windows; for the label-fine-tuned DINOv2 encoder only the head is refit) and
``<method>|original`` (the head as first trained). The elapsed-time baseline is refit per held-out group here.
"""

from __future__ import annotations

import numpy as np

from deadahead.baselines.elapsed_time import _fit_logistic, fit_predict, prepare
from deadahead.evaluation.aggregate import LEFT_OUT_FOLD, both_folds, outcome_text
from deadahead.evaluation.rows import CAUSES, GROUPS, TABLE1_HORIZON, align_scores, table1_rows
from deadahead.evaluation.table1 import table1, table1_unit

DEADAHEAD, VIDEO, DEEPHIT, TIME = "deadahead", "videomae_finetuned", "deephit", "elapsed_time"
COMPARATORS = ("dinov2_handcrafted", "dinov2", "dinov2_finetuned_handcrafted", "handcrafted")
POLICIES_ORDER = ("refit", "original")      # the Table 1 entry first, the original training beside it
METHOD_ORDER = (DEADAHEAD, VIDEO, DEEPHIT, *[f"{c}|{p}" for c in COMPARATORS for p in POLICIES_ORDER], TIME)
TIME_SEED = 0                               # the elapsed-time baseline has no seed (one deterministic fit per group)
EIGHT = "eight_folds_without_PNT1A_bp"
METHOD_LABELS = {
    DEADAHEAD: "DeadAhead (survival head on the fine-tuned video features and the handcrafted features)",
    VIDEO: "VideoMAE fine-tuned on the forecast labels",
    DEEPHIT: "DeepHit on the DeadAhead inputs",
    "dinov2_handcrafted": "DINOv2 image features + handcrafted features",
    "dinov2": "DINOv2 image features",
    "dinov2_finetuned_handcrafted": "label-fine-tuned DINOv2 features + handcrafted features",
    "handcrafted": "handcrafted features",
    TIME: "elapsed time in the movie (logistic)",
}


def prediction_scores(df, seed: int, rg) -> dict:
    """One method's held-out scores at one seed, per cause, aligned to the evaluated rows of its group."""
    out = {}
    for c in CAUSES:
        d = df[(df["cause"] == c) & (df["seed"] == seed)]
        if "horizon_min" in d and (d["horizon_min"] != 30).any():
            raise ValueError("a comparator row is not a 30-min row")
        out[c] = align_scores(d, d["score"].to_numpy(float), rg)
    return out


def method_scores(bundle: dict, rows, horizon: int = 30) -> tuple[dict, dict]:
    """{method: {(g, s): {cause: array over rows[rows.group == g]}}} for every method of the bundle and the
    elapsed-time baseline, and {method: reason} for a method that cannot cover the rows (never partially scored)."""
    out, failed = {}, {}

    def put(m, g, s, fn):
        if m in failed:
            return
        try:
            out.setdefault(m, {})[(g, s)] = fn()
        except ValueError as e:
            failed[m] = f"{g} s{s}: {e}"
            out.pop(m, None)
    elapsed_rows = prepare(bundle["rows"])
    for g in GROUPS:
        rg = rows[rows["group"] == g].reset_index(drop=True)
        for m in [m for m in METHOD_ORDER if m in bundle["predictions"]]:
            dg = bundle["predictions"][m]
            dg = dg[dg["group"] == g]
            for s in sorted(dg["seed"].unique()):
                put(m, g, int(s), lambda: prediction_scores(dg, int(s), rg))
        tr = elapsed_rows[(elapsed_rows["horizon_min"] == horizon) & (elapsed_rows["group"] != g)]
        te = rg.assign(time_in_movie_min=rg["t_obs_frame"].to_numpy(float) * 3.0)
        put(TIME, g, TIME_SEED, lambda: {c: fit_predict(tr, te, 2, c) for c in CAUSES})
    return out, failed


def elapsed_time_fit(bundle: dict, rows, horizon: int = TABLE1_HORIZON) -> dict:
    """The logistic fit of elapsed time, per cause, with and without the row weights: per held-out group the slope on
    the standardised time in the movie (fit on the other groups) and the AUPRC on ``rows``, with the macros."""
    base = prepare(bundle["rows"])
    base = base[base["horizon_min"] == horizon]
    held_out = prepare(rows)
    out = {}
    for c in CAUSES:
        slopes, scores = {"weighted": {}, "unweighted": {}}, {"weighted": {}, "unweighted": {}}
        for g in GROUPS:
            tr = base[base["group"] != g]
            rg = held_out[held_out["group"] == g]
            mu, sd = float(tr["time_in_movie_min"].mean()), float(tr["time_in_movie_min"].std()) or 1.0
            x = (tr["time_in_movie_min"].to_numpy() - mu) / sd
            x_held_out = (rg["time_in_movie_min"].to_numpy() - mu) / sd
            y = (tr["label"] == c).to_numpy(float)
            for name, w in (("weighted", tr["window_weight"].to_numpy(float)), ("unweighted", None)):
                a, b = _fit_logistic(np.column_stack([np.ones(len(x)), x]), y, w=w)
                slopes[name][g] = float(b)
                scores[name][g] = {TIME_SEED: table1_unit(a + b * x_held_out, rg["label"].to_numpy(), c)["auprc"]}
        out[c] = {name: {"slope_per_group": slopes[name],
                         "groups_slope_below_zero": int(sum(v < 0 for v in slopes[name].values())),
                         "groups_slope_above_zero": int(sum(v > 0 for v in slopes[name].values())),
                         "auprc": both_folds(scores[name])} for name in slopes}
    return out


def part_table1(bundle: dict, rows_filter=None) -> dict:
    """Table 1 on the common evaluation cohort and on the original rows, with the refit-minus-original and
    cohort-minus-original differences of the macros, the chance row and the elapsed-time fit with and without the row
    weights; ``rows_filter`` restricts both views' rows (sensitivity)."""
    tr = table1_rows(bundle["rows"], bundle["untyped"])
    out = {"counts": tr["counts"]}
    for view in ("cohort", "original"):
        rows = tr[view] if rows_filter is None else rows_filter(tr[view])
        sc, failed = method_scores(bundle, rows)
        res = table1(sc, rows, METHOD_ORDER)
        res["not_computed"] = failed
        res["n_rows"] = len(rows)
        out[view] = res
    out["refit_minus_original_policy"] = {"per_comparator": {}}
    for comp in COMPARATORS:
        for c in CAUSES:
            mc = out["cohort"]["methods"].get(f"{comp}|refit", {}).get(c)
            mo = out["cohort"]["methods"].get(f"{comp}|original", {}).get(c)
            if mc and mo:
                pc, po = mc["nine_folds"]["per_group"], mo["nine_folds"]["per_group"]
                gs = [g for g in GROUPS if g in pc and g in po]
                out["refit_minus_original_policy"]["per_comparator"][f"{comp}|{c}"] = {
                    "per_group": {g: pc[g] - po[g] for g in gs},
                    "macro": mc["nine_folds"]["macro"] - mo["nine_folds"]["macro"], "groups": gs}
    out["common_cohort_minus_original_cohort"] = {
        f"{m}|{c}": out["cohort"]["methods"][m][c]["nine_folds"]["macro"]
        - out["original"]["methods"][m][c]["nine_folds"]["macro"]
        for m in out["cohort"]["methods"] if m in out["original"]["methods"] for c in CAUSES}
    eight = tuple(g for g in GROUPS if g != LEFT_OUT_FOLD)
    out["chance"] = {c: {"nine_folds": chance_level(out["cohort"], c), EIGHT: chance_level(out["cohort"], c, eight)}
                     for c in CAUSES}
    out["elapsed_time_fit"] = elapsed_time_fit(bundle, tr["cohort"] if rows_filter is None
                                               else rows_filter(tr["cohort"]))
    return out


def chance_level(res: dict, cause: str, groups=GROUPS) -> float:
    """The mean over groups of the prevalence P / (P + Q + N) (the AUPRC of a random score)."""
    units = next(iter(res["methods"].values()))[cause]["units"]        # every method has the same counts
    prev = []
    for g in groups:
        u = next(v for k, v in units.items() if k.split("|")[0] == g)
        prev.append(u["P"] / (u["P"] + u["Q"] + u["N"]))
    return sum(prev) / len(prev)


def format_table1(t1: dict) -> str:
    """Table 1 as text: the macro AUPRC per method and cause, and DeadAhead's paired contrasts."""
    res = t1["cohort"]
    lines = [f"Table 1: AUPRC at 30 min, evaluation cohort ({res['n_rows']} rows, 9 held-out groups)", "",
             f"{'method':<42}{'lytic':>8}{'apoptosis':>11}{'lytic (8)':>11}{'apopt. (8)':>12}"]
    ch = t1["chance"]
    lines.append(f"{'chance (prevalence)':<42}{ch['LYTIC']['nine_folds']:>8.3f}{ch['APOPTOSIS']['nine_folds']:>11.3f}"
                 f"{ch['LYTIC'][EIGHT]:>11.3f}{ch['APOPTOSIS'][EIGHT]:>12.3f}")
    for m in METHOD_ORDER:
        if m not in res["methods"]:
            continue
        b = res["methods"][m]
        lines.append(f"{m:<42}{b['LYTIC']['nine_folds']['macro']:>8.3f}{b['APOPTOSIS']['nine_folds']['macro']:>11.3f}"
                     f"{b['LYTIC']['eight_folds_without_PNT1A_bp']['macro']:>11.3f}"
                     f"{b['APOPTOSIS']['eight_folds_without_PNT1A_bp']['macro']:>12.3f}")
    lines += ["", "(8): the eight groups other than PNT1A_bp; |refit: comparator head refit without the untyped-death "
              "training windows; |original: as first trained"]
    lines += [f"  {k}: {v}" for k, v in METHOD_LABELS.items()]
    lines += ["",
              "DeadAhead minus each method: macro difference, groups ahead of 9, outcome"]
    for m in METHOD_ORDER[1:]:
        for c in ("LYTIC", "APOPTOSIS"):
            k = f"{DEADAHEAD} - {m}|{c}"
            if k in res["contrasts"]:
                p = res["contrasts"][k]["nine_folds"]
                lines.append(f"  {m:<40}{c:<11}{p.get('macro_difference', float('nan')):+8.3f}"
                             f"{p.get('x_ahead', 0):>4d}/{p['n']}  {outcome_text(p)}")
    if res["not_computed"]:
        lines += ["", f"not computed: {res['not_computed']}"]
    return "\n".join(lines + ["", format_elapsed_time_fit(t1["elapsed_time_fit"])])


def format_elapsed_time_fit(fit: dict) -> str:
    """The elapsed-time fit as text: the slope per held-out group with and without the row weights, and the AUPRC."""
    lines = ["Elapsed time's logistic fit: slope on the standardised time in the movie per held-out group, and macro "
             "AUPRC, with the row weights (Table 1) and without them",
             f"{'held-out group':<16}{'lytic':>10}{'unweighted':>12}{'apoptosis':>11}{'unweighted':>12}"]
    columns = [(c, n) for c in ("LYTIC", "APOPTOSIS") for n in ("weighted", "unweighted")]
    widths = (10, 12, 11, 12)
    for g in GROUPS:
        cells = "".join(f"{fit[c][n]['slope_per_group'][g]:>+{w}.3f}" for (c, n), w in zip(columns, widths))
        lines.append(f"{g:<16}{cells}")
    for label, key in (("slope below 0", "groups_slope_below_zero"), ("slope above 0", "groups_slope_above_zero")):
        cells = "".join(f"{fit[c][n][key]:>{w}d}" for (c, n), w in zip(columns, widths))
        lines.append(f"{label:<16}{cells}")
    cells = "".join(f"{fit[c][n]['auprc']['nine_folds']['macro']:>{w}.3f}" for (c, n), w in zip(columns, widths))
    lines.append(f"{'macro AUPRC':<16}{cells}")
    return "\n".join(lines)
