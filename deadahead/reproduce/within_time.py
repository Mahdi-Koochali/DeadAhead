"""Reproduce the post hoc within-hour scoring: the 30-min ranking when a forecast competes only with forecasts of the
same group in the same hour of imaging.

Rows: Table 1's evaluation cohort at the sensitivity part's common cut, t + 10 <= 438, the same for every method. Bands
of the forecast frame t: 1 hour (primary), 2 hours and 30 min (these two decide nothing). Per method, route, group and
seed, each band with a death by the route is scored on its own by Table 1's AUPRC from the stored 30-min score, and the
band values are averaged with the deaths as weights; then the mean over seeds and the macro over groups, as in Table 1.
A random order reaches the exact expected AUPRC E(P, n), above the positive rate on few rows, so these values are on
their own scale and are not compared with Table 1. Elapsed time is tied within a forecast time, so it is set beside the
positive-rate chance. A second design ranks each score within its band, then takes one AUPRC over the group; it decides
nothing. Its ceiling is the value of an untied score that puts every death first in its band.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from deadahead.evaluation.aggregate import AHEAD_X, both_folds, outcome_text, paired_both
from deadahead.evaluation.alarms import rows_id
from deadahead.evaluation.rows import CAUSES, GROUPS, key_index, set_counts, table1_rows, window_sets
from deadahead.evaluation.table1 import table1_unit
from deadahead.reproduce.gap import _per_seed
from deadahead.reproduce.gap_window import compare_block
from deadahead.reproduce.sensitivity import cut30
from deadahead.reproduce.table1 import DEADAHEAD, DEEPHIT, TIME, TIME_SEED, VIDEO, chance_level, method_scores

FIRST_FRAME, LAST_FRAME = 40, 420            # the forecast frames of the rows
EDGES = {"1h": tuple(range(40, 421, 20)), "2h": tuple(range(40, 401, 40)), "30min": tuple(range(40, 421, 10))}
BANDS = tuple(EDGES)                         # the primary first
PRIMARY_BANDS = "1h"
IMAGE = "dinov2_handcrafted|refit"
METHOD_NAMES = {DEADAHEAD: "DeadAhead", VIDEO: "VideoMAE (fine-tuned)", DEEPHIT: "DeepHit", IMAGE: "DINOv2 + handcrafted",
                "dinov2_finetuned_handcrafted|refit": "Fine-tuned DINOv2 + handcrafted",
                "handcrafted|refit": "Handcrafted", "dinov2|refit": "DINOv2", TIME: "elapsed time"}
METHODS = tuple(METHOD_NAMES)
OTHERS = METHODS[1:]                         # each contrast is DeadAhead minus one of these
PRIMARY_CONTRAST = f"{DEADAHEAD} - {IMAGE}"
ROUTE_SUFFIX = {"LYTIC": "L", "APOPTOSIS": "A"}
COMMON_CUT_BLOCK = "4.1_table1_common_cut_30min"     # the sensitivity part's Table 1 at the same rows
CHECK_TOL = 1e-12                            # the cross-checks with the sensitivity part
BAND_ROLE = {"1h": "primary: 20 frames wide, 1 hour", "2h": "secondary, deciding nothing: 40 frames wide, 2 hours",
             "30min": "secondary, deciding nothing: 10 frames wide, 30 min, one forecast time per band"}
CHANCE_RULE = {"random_order": "E(P, n) = (1/n) [ (P - 1)/(n - 1) (n - H_n) + H_n ] per band, n = 1 gives 1, weighted "
                               "by P_b within a group, then the macro over the groups",
               "positive_rate": "the P_b-weighted mean of P_b / n_b within a group, then the macro over the groups: "
                                "what a score tied across a band gets from auprc"}
WITHIN_RULE = ("per method, route, group and seed, Table 1's AUPRC of each band with at least one death by the route, "
               "over that band's rows, from the stored 30-min score; the mean of these weighted by the deaths of each "
               "band; the mean over seeds within a group, then the macro over the groups")
POOLED_RULE = ("within each 1-hour band of a group a score becomes (r - 1) / (n - 1), r its average rank, n the rows of "
               "the band, a band of one row 0.5; one AUPRC over the group's rows. Secondary, deciding nothing; not on "
               "Table 1's scale")
CEILING_RULE = "an untied score that puts every death by the route first in its band, labels only"
TIED_LABEL_RULE = "the label itself as the score, its ties kept; deciding nothing"


# ------------------------------------------------------------------ bands and chance
def band_index(t, edges):
    """The band of each row: numpy.digitize(t, edges). 0 is below the first edge; the last band is open above."""
    return np.digitize(np.asarray(t), np.asarray(edges))


def one_band(n: int):
    """One band holding every row."""
    return np.zeros(int(n), dtype=int)


def positive_weighted(values, weights) -> float:
    """The mean of the band values weighted by their deaths; nan when no band holds one."""
    total = sum(weights)
    if total == 0:
        return float("nan")
    return math.fsum(w * v for v, w in zip(values, weights)) / total


def harmonic(n: int) -> float:
    """H_n, the n-th harmonic number."""
    return math.fsum(1.0 / i for i in range(1, int(n) + 1))


def expected_auprc(P: int, n: int) -> float:
    """E(P, n), the exact expected AUPRC of a random order of n rows holding P positives; for n = 1 it is 1."""
    P, n = int(P), int(n)
    if P < 1 or n < P:
        raise ValueError(f"E(P, n) needs 1 <= P <= n, got P = {P}, n = {n}")
    if n == 1:
        return 1.0
    h = harmonic(n)
    return ((P - 1) / (n - 1) * (n - h) + h) / n


def band_counts(label, bands, cause: str) -> dict:
    """{band: {"P": deaths by the route, "n": rows in P, Q or N}}."""
    sets = window_sets(label, cause)
    b = np.asarray(bands)
    out = {}
    for u in np.unique(b):
        m = b == u
        p = int((sets["P"] & m).sum())
        out[int(u)] = {"P": p, "n": p + int((sets["Q"] & m).sum()) + int((sets["N"] & m).sum())}
    return out


def band_table(label, bands, cause: str) -> dict:
    """The band counts as three lists."""
    cnt = band_counts(label, bands, cause)
    return {"band": list(cnt), "P": [v["P"] for v in cnt.values()], "n": [v["n"] for v in cnt.values()]}


def band_chance(label, bands, cause: str) -> dict:
    """One group's chance: E(P, n) and P / n weighted by P, over the bands with a death."""
    cnt = [v for v in band_counts(label, bands, cause).values() if v["P"] > 0]
    w = [v["P"] for v in cnt]
    return {"random_order": positive_weighted([expected_auprc(v["P"], v["n"]) for v in cnt], w),
            "positive_rate": positive_weighted([v["P"] / v["n"] for v in cnt], w),
            "positives": sum(w), "rows_that_count": sum(v["n"] for v in cnt), "bands_counted": len(cnt)}


# ------------------------------------------------------------------ the two designs
def within_band_value(score, label, bands, cause: str) -> float:
    """One unit: each band with a death scored on its own, the mean weighted by the deaths; other bands add nothing."""
    s, lab, b = np.asarray(score, float), np.asarray(label), np.asarray(bands)
    if not s.shape == lab.shape == b.shape:
        raise ValueError(f"score, label and bands differ in shape: {s.shape}, {lab.shape}, {b.shape}")
    values, weights = [], []
    for u in np.unique(b):
        m = b == u
        v = table1_unit(s[m], lab[m], cause)
        if v["P"] > 0:
            values.append(v["auprc"])
            weights.append(v["P"])
    return positive_weighted(values, weights)


def rank_rule(score, bands):
    """Within each band each score becomes (r - 1) / (n - 1), r its average rank and n the rows of the band; a band of
    one row gets 0.5, and so does every row of a band tied throughout."""
    s, b = np.asarray(score, float), np.asarray(bands)
    if s.shape != b.shape:
        raise ValueError(f"score and bands differ in shape: {s.shape}, {b.shape}")
    if not np.all(np.isfinite(s)):
        raise ValueError("a non-finite score cannot be ranked")
    out = np.full(s.shape, 0.5)
    for u in np.unique(b):
        m = b == u
        n = int(m.sum())
        if n > 1:
            out[m] = (pd.Series(s[m]).rank(method="average").to_numpy() - 1.0) / (n - 1)
    return out


def pooled_value(score, label, bands, cause: str) -> float:
    """One unit of the second design: the rank rule, then one AUPRC over the group's rows."""
    return table1_unit(rank_rule(score, bands), label, cause)["auprc"]


def deaths_first_score(label, cause: str):
    """A score that puts every death by the route above every other row, with no two rows tied."""
    y = window_sets(label, cause)["P"]
    return 2.0 * y + np.arange(y.size) / max(y.size, 1)


def ceiling_value(label, bands, cause: str) -> float:
    """The second design's value for ``deaths_first_score``."""
    return pooled_value(deaths_first_score(label, cause), label, bands, cause)


def tied_label_value(label, bands, cause: str) -> float:
    """The second design's value for the label itself as the score, its ties kept."""
    return pooled_value(window_sets(label, cause)["P"].astype(float), label, bands, cause)


# ------------------------------------------------------------------ rows and units
def scored_rows(bundle: dict) -> tuple:
    """Table 1's cohort at the common cut, and what the cut left out."""
    tr = table1_rows(bundle["rows"], bundle["untyped"])
    cohort = tr["cohort"]
    rows = cut30(cohort)
    gone = cohort[~key_index(cohort).isin(key_index(rows))]
    return rows, {"cohort_rows": len(cohort), "cohort_removed": tr["counts"]["cohort_removed"],
                  "U_excluded": tr["counts"]["U_excluded"],
                  "left_out": {"n": len(gone), "frames": sorted(int(f) for f in gone["t_obs_frame"].unique()),
                               "labels": {str(k): int(v) for k, v in gone["label"].value_counts().items()}}}


def context(rows) -> dict:
    """The rows, per group the frames and labels, and each row's band at every width."""
    by = {}
    for g in GROUPS:
        rg = rows[(rows["group"] == g).to_numpy()]
        by[g] = {"t": rg["t_obs_frame"].to_numpy(np.int64), "label": rg["label"].to_numpy().astype(str)}
    return {"rows": rows, "by_group": by,
            "bands": {name: {g: band_index(v["t"], EDGES[name]) for g, v in by.items()} for name in BANDS}}


def scored_units(ctx: dict, scores: dict, value, bands: dict | None = None) -> dict:
    """{method: {route: {group: {seed: value(score, label, bands, route)}}}}; one band for every row without ``bands``."""
    out = {}
    for m, by_gs in scores.items():
        for (g, s), sc in sorted(by_gs.items()):
            lab = ctx["by_group"][g]["label"]
            b = one_band(lab.size) if bands is None else bands[g]
            for c in CAUSES:
                out.setdefault(m, {}).setdefault(c, {}).setdefault(g, {})[s] = value(sc[c], lab, b, c)
    return out


# ------------------------------------------------------------------ the checks
def check_bands(ctx: dict) -> dict:
    """Every row of the 9 groups in a band (frames 40 to 420), and the band counts summing to the rows, at every width."""
    probs = []
    outside = sorted(set(ctx["rows"]["group"].astype(str)) - set(GROUPS))
    if outside:
        probs.append(f"rows of groups outside the 9: {outside}")
    rows_in_bands = {}
    for name in BANDS:
        total = 0
        for g, v in ctx["by_group"].items():
            t, b = v["t"], ctx["bands"][name][g]
            low, high = int((t < FIRST_FRAME).sum()), int((t > LAST_FRAME).sum())
            if low or high:
                probs.append(f"{name} {g}: {low} row(s) below frame {FIRST_FRAME}, {high} above {LAST_FRAME}")
            total += int(np.bincount(b[(t >= FIRST_FRAME) & (t <= LAST_FRAME)], minlength=len(EDGES[name]) + 1).sum())
        rows_in_bands[name] = total
        if total != len(ctx["rows"]):
            probs.append(f"{name}: the band counts sum to {total}, the row count is {len(ctx['rows'])}")
    return {"n_rows": len(ctx["rows"]), "rows_in_bands": rows_in_bands, "problems": probs, "passes": not probs}


def check_one_band(units: dict, block: dict, tol: float = CHECK_TOL) -> dict:
    """With one band, every method's per-group and macro value equals the sensitivity part's Table 1 at the same rows."""
    probs, n = [], 0
    for m, by_c in units.items():
        for c, pgs in by_c.items():
            n += 1
            want = ((block.get("methods") or {}).get(m) or {}).get(c)
            probs += [f"{m} {c}: {p}" for p in compare_block(both_folds(pgs), want, tol)]
    return {"n_checks": n, "tolerance": tol, "problems": probs[:10], "n_problems": len(probs),
            "passes": n > 0 and not probs}


def check_positive_rate(ctx: dict, block: dict | None, tol: float = CHECK_TOL) -> dict:
    """With one band, the positive-rate chance is the macro of |P| / (|P| + |Q| + |N|) from the rows, and with the
    sensitivity part it equals that part's chance from its own counts."""
    probs, out = [], {}
    for c in CAUSES:
        rate = {}
        for g, v in ctx["by_group"].items():
            n = v["label"].size
            k = set_counts(window_sets(v["label"], c))
            if k["P"] + k["Q"] + k["N"] != n:
                probs.append(f"{c} {g}: a row outside P, Q and N")
            rate[g] = k["P"] / n if n else 0.0
            if k["P"] and not abs(band_chance(v["label"], one_band(n), c)["positive_rate"] - rate[g]) <= tol:
                probs.append(f"{c} {g}: the one-band chance is not |P| / (|P| + |Q| + |N|)")
        macro = float(np.mean([rate[g] for g in GROUPS]))
        d = {"from_the_rows": macro}
        if block is not None:
            d["from_the_sensitivity_part"] = chance_level(block, c)
            if not abs(macro - d["from_the_sensitivity_part"]) <= tol:
                probs.append(f"{c}: {macro} from the rows, {d['from_the_sensitivity_part']} in the sensitivity part")
        out[c] = d
    return {"tolerance": tol, "routes": out, "problems": probs, "passes": not probs}


def check_elapsed_time(ctx: dict, scores: dict, tol: float = CHECK_TOL) -> dict:
    """Elapsed time is tied within a forecast time, so under 30-min bands it equals the positive-rate chance."""
    bands = ctx["bands"]["30min"]
    units = scored_units(ctx, {TIME: scores[TIME]}, within_band_value, bands)[TIME]
    probs, worst, n = [], 0.0, 0
    for c in CAUSES:
        for g, by_seed in units[c].items():
            a, b = by_seed[TIME_SEED], band_chance(ctx["by_group"][g]["label"], bands[g], c)["positive_rate"]
            if math.isnan(a) and math.isnan(b):
                continue
            n += 1
            d = abs(a - b) if not (math.isnan(a) or math.isnan(b)) else math.inf
            worst = max(worst, d)
            if not d <= tol:
                probs.append(f"{c} {g}: elapsed time differs from the positive-rate chance by {d:.3g}")
    return {"n_checks": n, "tolerance": tol, "max_abs_diff": worst, "problems": probs[:10], "passes": not probs}


def check_rank_rule(raw: dict, ranked: dict, tol: float = CHECK_TOL) -> dict:
    """With one band, the rank rule leaves every unit's AUPRC at its raw value (ties keep their order)."""
    probs, worst, n = [], 0.0, 0
    for m, by_c in raw.items():
        for c, by_g in by_c.items():
            for g, by_s in by_g.items():
                for s, v in by_s.items():
                    a = ranked[m][c][g][s]
                    if math.isnan(v) and math.isnan(a):
                        continue
                    n += 1
                    d = abs(v - a) if not (math.isnan(v) or math.isnan(a)) else math.inf
                    worst = max(worst, d)
                    if not d <= tol:
                        probs.append(f"{m} {c} {g} s{s}: the ranked value differs by {d:.3g}")
    return {"n_checks": n, "tolerance": tol, "max_abs_diff": worst, "problems": probs[:10], "passes": n > 0 and not probs}


def _raise_on_failure(checks: dict) -> None:
    failed = [k for k, v in checks.items() if not v["passes"]]
    if failed:
        raise ValueError(f"the within-hour scoring failed a check: {failed}: "
                         + "; ".join(p for k in failed for p in checks[k]["problems"][:2]))


# ------------------------------------------------------------------ the blocks of the output
def chance_block(chances: dict) -> dict:
    """Both chances, per group and macro; ``chances`` {group: band_chance}."""
    return {k: {"rule": CHANCE_RULE[k], **both_folds({g: {TIME_SEED: v[k]} for g, v in chances.items()})}
            for k in CHANCE_RULE}


def method_entry(m: str, pgs: dict, chances: dict | None = None) -> dict:
    """One method's values at one band width and route, with the chance beside it over the same groups: the
    positive-rate chance for elapsed time, the random-order chance for every other method."""
    both = both_folds(pgs)
    out = {"name": METHOD_NAMES[m], "per_seed": _per_seed(pgs), **both}
    if chances is not None:
        keep = both["nine_folds"]["groups_contributing"]
        kind = "positive_rate" if m == TIME else "random_order"
        out["chance_beside"] = {"kind": kind, "n_groups": len(keep), "groups": keep,
                                "macro": float(np.mean([chances[g][kind] for g in keep])) if keep else float("nan")}
    return out


def contrast_block(methods: dict, primary: bool) -> dict:
    """DeadAhead minus each other method, by the function Table 1 uses."""
    out = {}
    x = methods[DEADAHEAD]["nine_folds"]["per_group"]
    for m in OTHERS:
        if m not in methods:
            continue
        key = f"{DEADAHEAD} - {m}"
        k = paired_both(x, methods[m]["nine_folds"]["per_group"])
        k["deadahead_called_ahead"] = k["nine_folds"].get("outcome") == AHEAD_X
        k["decides"] = "the primary contrast" if primary and key == PRIMARY_CONTRAST else "nothing"
        out[key] = k
    return out


def width_block(ctx: dict, scores: dict, name: str) -> dict:
    """One band width: per route the band counts, both chances, every method and the contrasts."""
    bands = ctx["bands"][name]
    units = scored_units(ctx, scores, within_band_value, bands)
    out = {}
    for c in CAUSES:
        chances = {g: band_chance(v["label"], bands[g], c) for g, v in ctx["by_group"].items()}
        methods = {m: method_entry(m, units[m][c], chances) for m in METHODS if m in units}
        keys = ("positives", "rows_that_count", "bands_counted")
        out[c] = {"counts": {"per_group": {g: {**{k: chances[g][k] for k in keys},
                                               "bands": band_table(ctx["by_group"][g]["label"], bands[g], c)}
                                           for g in GROUPS},
                             "rows_that_count": sum(chances[g]["rows_that_count"] for g in GROUPS),
                             "positives": sum(chances[g]["positives"] for g in GROUPS)},
                  "chance": chance_block(chances), "methods": methods,
                  "contrasts": contrast_block(methods, primary=name == PRIMARY_BANDS)}
    return out


def pooled_block(ctx: dict, scores: dict) -> dict:
    """The second design at 1-hour bands: per route the ceiling, every method and the contrasts."""
    bands = ctx["bands"][PRIMARY_BANDS]
    units = scored_units(ctx, scores, pooled_value, bands)
    out = {}
    for c in CAUSES:
        ceiling = {g: {TIME_SEED: ceiling_value(v["label"], bands[g], c)} for g, v in ctx["by_group"].items()}
        tied = {g: {TIME_SEED: tied_label_value(v["label"], bands[g], c)} for g, v in ctx["by_group"].items()}
        methods = {m: method_entry(m, units[m][c]) for m in METHODS if m in units}
        out[c] = {"ceiling": {"rule": CEILING_RULE, **both_folds(ceiling)},
                  "label_as_a_tied_score": {"rule": TIED_LABEL_RULE, **both_folds(tied)}, "methods": methods,
                  "contrasts": contrast_block(methods, primary=False)}
    return out


def _contrast_summary(block: dict) -> dict:
    out = {}
    for k, v in block.items():
        nf = v["nine_folds"]
        out[k] = {"macro_difference": nf.get("macro_difference"), "x_ahead": nf.get("x_ahead"), "n": nf.get("n"),
                  "outcome": nf.get("outcome"), "deadahead_called_ahead": v["deadahead_called_ahead"],
                  "decides": v["decides"]}
    return out


def summary(within: dict, pooled: dict) -> dict:
    """A compact table: every macro, both chances and every contrast's outcome."""
    out = {"within_band": {}, "pooled_1h": {}}
    for name, routes in within.items():
        for c, blk in routes.items():
            out["within_band"].setdefault(name, {})[c] = {
                "chance_random_order": blk["chance"]["random_order"]["nine_folds"]["macro"],
                "chance_positive_rate": blk["chance"]["positive_rate"]["nine_folds"]["macro"],
                "methods": {m: {"macro": e["nine_folds"]["macro"], "n_groups": e["nine_folds"]["n_groups"],
                                "macro_8_groups": e["eight_folds_without_PNT1A_bp"]["macro"],
                                "chance_beside": e["chance_beside"]["kind"],
                                "chance_beside_macro": e["chance_beside"]["macro"]}
                            for m, e in blk["methods"].items()},
                "contrasts": _contrast_summary(blk["contrasts"])}
    for c, blk in pooled.items():
        out["pooled_1h"][c] = {"ceiling": blk["ceiling"]["nine_folds"]["macro"],
                               "label_as_a_tied_score": blk["label_as_a_tied_score"]["nine_folds"]["macro"],
                               "methods": {m: {"macro": e["nine_folds"]["macro"], "n_groups": e["nine_folds"]["n_groups"]}
                                           for m, e in blk["methods"].items()},
                               "contrasts": _contrast_summary(blk["contrasts"])}
    return out


def sentence_values(within: dict) -> dict:
    """The values of the paper's sentence and each route's decision-rule outcome."""
    values, outcomes = {}, {}
    for c in CAUSES:
        k = ROUTE_SUFFIX[c]
        blk = within[PRIMARY_BANDS][c]
        con = blk["contrasts"][PRIMARY_CONTRAST]
        values[f"x_{k}"] = blk["methods"][DEADAHEAD]["nine_folds"]["macro"]
        values[f"y_{k}"] = blk["chance"]["random_order"]["nine_folds"]["macro"]
        values[f"z_{k}"] = blk["methods"][IMAGE]["nine_folds"]["macro"]
        values[f"d_{k}"] = con["nine_folds"].get("macro_difference")
        values[f"k_{k}"] = con["nine_folds"].get("x_ahead")
        outcomes[c] = {"outcome": con["nine_folds"].get("outcome"), "n_groups": con["nine_folds"].get("n"),
                       "deadahead_called_ahead": con["deadahead_called_ahead"]}
    return {"rule": "x DeadAhead, y a random order, z DINOv2 + handcrafted, d the lead, k the groups ahead, at 1-hour "
                    "bands; L lytic, A apoptosis", "values": values, "outcomes": outcomes}


# ------------------------------------------------------------------ the part
def part_within_time(bundle: dict, sensitivity: dict | None = None) -> dict:
    """Every method at every band width and the second design, on the rows of the common cut; with ``sensitivity``
    (that part's result) the one-band values and the chance are checked against it."""
    rows, info = scored_rows(bundle)
    ctx = context(rows)
    block = (sensitivity or {}).get(COMMON_CUT_BLOCK, {}).get("cohort")
    checks = {"every_row_in_a_band": check_bands(ctx), "positive_rate_with_one_band": check_positive_rate(ctx, block)}
    _raise_on_failure(checks)
    all_scores, failed = method_scores(bundle, rows)
    scores = {m: all_scores[m] for m in METHODS if m in all_scores}
    not_computed = {m: failed.get(m, "not in the bundle") for m in METHODS if m not in scores}
    lacking = [f"{m}: {not_computed[m]}" for m in (DEADAHEAD, IMAGE) if m in not_computed]
    if lacking:
        raise ValueError(f"the within-hour scoring needs DeadAhead and DINOv2 + handcrafted ({'; '.join(lacking)})")
    raw = scored_units(ctx, scores, within_band_value)
    checks["rank_rule_with_one_band"] = check_rank_rule(raw, scored_units(ctx, scores, pooled_value))
    if TIME in scores:
        checks["elapsed_time_under_30_min_bands"] = check_elapsed_time(ctx, scores)
    if block is not None:
        checks["one_band_equals_the_sensitivity_part"] = check_one_band(raw, block)
    _raise_on_failure(checks)
    within = {name: width_block(ctx, scores, name) for name in BANDS}
    pooled = pooled_block(ctx, scores)
    return {"post_hoc": True,
            "reading": "descriptive; on its own scale, not compared with Table 1; no p-value",
            "rows": {"rule": "Table 1's cohort at the common cut, t + 10 <= 438; the same rows and labels for every "
                             "method", "n_rows": len(rows), "rows_id": rows_id(rows), **info},
            "bands": {name: {"edges": list(EDGES[name]), "role": BAND_ROLE[name]} for name in BANDS},
            "methods": {m: METHOD_NAMES[m] for m in METHODS}, "not_computed": not_computed,
            "statistic": {"within_band": WITHIN_RULE, "pooled": POOLED_RULE, "primary_bands": PRIMARY_BANDS,
                          "primary_contrast": PRIMARY_CONTRAST},
            "within_band": within, "pooled": pooled, "summary": summary(within, pooled),
            "paper_sentence_values": sentence_values(within), "checks": checks}


# ------------------------------------------------------------------ text
def _lead(k: dict | None) -> str:
    if k is None:
        return ""
    if k["macro_difference"] is None:
        return k["outcome"]
    mark = "*" if k["deadahead_called_ahead"] else ""
    return f"{k['macro_difference']:+.3f} {k['x_ahead']}/{k['n']}{mark}"


def format_within_time(res: dict) -> str:
    """The within-hour scoring as text: every method at 1-hour bands, the primary contrast at each band width, and the
    second design with its ceiling."""
    w = res["summary"]["within_band"]
    lines = ["Post hoc within-hour scoring: AUPRC at 30 min with each hour of imaging that held a death scored on its own "
             "(nine-group macro; its own scale, not Table 1's)", "",
             f"{'method':<42}{'lytic':>8}{'lead':>15}{'apoptosis':>11}{'lead':>15}"]
    lyt, apo = w[PRIMARY_BANDS]["LYTIC"], w[PRIMARY_BANDS]["APOPTOSIS"]
    for m in METHODS:
        if m not in lyt["methods"]:
            continue
        cells = []
        for b in (lyt, apo):
            k = b["contrasts"].get(f"{DEADAHEAD} - {m}")
            cells.append((b["methods"][m]["macro"], _lead(k)))
        lines.append(f"{m:<42}{cells[0][0]:>8.3f}{cells[0][1]:>15}{cells[1][0]:>11.3f}{cells[1][1]:>15}")
    lines += [f"{'a random order':<42}{lyt['chance_random_order']:>8.3f}{'':>15}{apo['chance_random_order']:>11.3f}",
              f"{'the positive rate (beside elapsed time)':<42}{lyt['chance_positive_rate']:>8.3f}{'':>15}"
              f"{apo['chance_positive_rate']:>11.3f}",
              "lead: DeadAhead minus the method, groups ahead of 9; * meets the decision rule", ""]
    for c, name in (("LYTIC", "lytic"), ("APOPTOSIS", "apoptosis")):
        b, k = w[PRIMARY_BANDS][c], w[PRIMARY_BANDS][c]["contrasts"][PRIMARY_CONTRAST]
        lines.append(f"Post hoc within-hour AUPRC, {name}: DeadAhead {b['methods'][DEADAHEAD]['macro']:.3f}, DINOv2 + "
                     f"handcrafted {b['methods'][IMAGE]['macro']:.3f}, a random order {b['chance_random_order']:.3f}; lead "
                     f"{k['macro_difference']:+.3f}, ahead in {k['x_ahead']} of {k['n']} groups, {outcome_text(k)}")
    lines += ["", "Other band widths, DeadAhead against DINOv2 + handcrafted (these decide nothing)",
              f"{'bands':<9}{'route':<11}{'DeadAhead':>10}{'image':>8}{'random order':>14}{'lead':>15}"]
    for name in BANDS[1:]:
        for c in ("LYTIC", "APOPTOSIS"):
            b = w[name][c]
            lines.append(f"{name:<9}{c.lower():<11}{b['methods'][DEADAHEAD]['macro']:>10.3f}"
                         f"{b['methods'][IMAGE]['macro']:>8.3f}{b['chance_random_order']:>14.3f}"
                         f"{_lead(b['contrasts'][PRIMARY_CONTRAST]):>15}")
    p = res["summary"]["pooled_1h"]
    lines += ["", ("A second design, deciding nothing: scores ranked within each hour of a group, then one AUPRC over the "
                   "group; ceiling: an untied score with every death first in its hour"),
              f"{'route':<11}{'DeadAhead':>10}{'image':>8}{'lead':>15}{'ceiling':>9}"]
    for c in ("LYTIC", "APOPTOSIS"):
        b = p[c]
        lines.append(f"{c.lower():<11}{b['methods'][DEADAHEAD]['macro']:>10.3f}{b['methods'][IMAGE]['macro']:>8.3f}"
                     f"{_lead(b['contrasts'][PRIMARY_CONTRAST]):>15}{b['ceiling']:>9.3f}")
    return "\n".join(lines)
