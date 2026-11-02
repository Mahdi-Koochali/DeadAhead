"""Reproduce the post hoc alarm-rate analysis: the share of deaths warned against the alarm rate, and one comparison
at a matched achieved alarm rate.

The thresholds are set exactly as in the warnings and image-warnings parts, on the training groups only, with only the
target rate changed: 0.05 to 0.80 alarms per event-free reference cell-hour in steps of 0.05 for DeadAhead, the image
baseline and elapsed time, and 0.005 to 0.800 in steps of 0.005 for DeadAhead and the image baseline. The target c* is
the image baseline's achieved pooled lytic rate at 0.2; DeadAhead's matched target is the grid rate whose achieved
pooled lytic rate is closest to c* (ties to the lower achieved rate, then the lower target).
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from deadahead.evaluation.aggregate import outcome_text, paired_both
from deadahead.evaluation.alarms import (NOMINAL, PRIMARY_RATE, ROW_HOURS, threshold_and_apply, warning_summary,
                                         warning_unit)
from deadahead.evaluation.rows import CAUSES, GROUPS, SEEDS
from deadahead.reproduce.compare import compare_tree
from deadahead.reproduce.image_warnings import (IMAGE, IMAGE_LABEL, LAST_HOUR, image_ref_scores,
                                                index_image_predictions, roundtrip)
from deadahead.reproduce.table1 import prediction_scores
from deadahead.reproduce.warning_times import (DEADAHEAD, TIME, TIME_SEED, index_predictions, inner_ref_scores,
                                          outer_scores, time_for, warning_population)

SWEEP_STEPS, SWEEP_DEN = 16, 20               # r = j / 20, j = 1..16: 0.05, 0.10, ..., 0.80
GRID_STEPS, GRID_DEN = 160, 200               # r = k / 200, k = 1..160: 0.005, 0.010, ..., 0.800
CHECK_RATES = ("0.2_per_cell_hour", "0.4_per_cell_hour")     # the rates of the warnings and image-warnings parts
ALARM_METHODS = (DEADAHEAD, IMAGE, TIME)
GRID_METHODS = (DEADAHEAD, IMAGE)
LYTIC, APOPTOSIS = "LYTIC", "APOPTOSIS"
CHECK_TOL = 1e-12                             # the cross-checks with the warnings and image-warnings parts
GRID_TOL = 0.0                                # the grid equals the sweep at every rate both hold
METHOD_LABELS = {DEADAHEAD: "DeadAhead with its training-only thresholds (216 cross-fitted inner heads)",
                 IMAGE: IMAGE_LABEL, TIME: "elapsed time: the one-draw time score, used as it is"}
DIFFERENCE = "difference_deadahead_minus_image_last_60_min"


# ------------------------------------------------------------------ the rates
def rate_name(r: Fraction) -> str:
    """'<r>_per_cell_hour' with r as a short decimal (0.2 -> '0.2_per_cell_hour')."""
    r = Fraction(r)
    n = r * 1000
    if n.denominator != 1:
        raise ValueError(f"rate {r} is not on a 0.001 grid")
    return f"{n.numerator / 1000:.3f}".rstrip("0").rstrip(".") + "_per_cell_hour"


def alpha_of(r: Fraction) -> Fraction:
    """The row fraction of a rate per cell-hour: alpha = r x ROW_HOURS (0.5 h per reference row), exact."""
    return Fraction(r) * Fraction(ROW_HOURS)


def rates(steps: int, den: int) -> dict:
    """{name: (r, alpha)} for r = j / den, j = 1..steps."""
    out = {}
    for j in range(1, steps + 1):
        r = Fraction(j, den)
        out[rate_name(r)] = (r, alpha_of(r))
    return out


def sweep_rates() -> dict:
    return rates(SWEEP_STEPS, SWEEP_DEN)


def grid_rates() -> dict:
    return rates(GRID_STEPS, GRID_DEN)


# ------------------------------------------------------------------ rows, scores, one rate
def warning_inputs(bundle: dict, rows) -> dict:
    """Per method and (g, s, route): (reference scores, held-out scores), read once by the calls of the warnings and
    image-warnings parts; nothing here depends on the rate."""
    neg = rows["negative"].to_numpy(bool)
    ts = time_for(rows)
    out = {m: {} for m in ALARM_METHODS}
    rg_by = {}
    for g in GROUPS:
        gm = (rows["group"] == g).to_numpy()
        rg = rows[gm].reset_index(drop=True)
        rg_by[g] = rg
        for c in CAUSES:
            for s in SEEDS:
                ref = inner_ref_scores(bundle, g, s, rows, neg, c)
                out[DEADAHEAD][(g, s, c)] = (ref, outer_scores(bundle, g, s, c, rg))
                try:
                    ref_i = image_ref_scores(bundle, g, s, rows, neg, c)
                    te_i = prediction_scores(bundle["img_outer"][g], s, rg)[c]
                except ValueError as e:
                    raise ValueError(f"{IMAGE} {g} s{s} {c}: {e} (never partially scored)") from None
                out[IMAGE][(g, s, c)] = (ref_i, te_i)
            out[TIME][(g, TIME_SEED, c)] = (np.asarray(ts)[(~gm) & neg], np.asarray(ts)[gm])
    return {"rows": rg_by, "scores": out}


def threshold_key(m: str, g: str, s: int, c: str, rate: str) -> str:
    """The thresholds' keys of the warnings and image-warnings parts."""
    return f"{m}|{g}|{c}|{rate}" if m == TIME else f"{m}|{g}|s{s}|{c}|{rate}"


def rate_pass(wi: dict, m: str, rate: str, a: Fraction) -> tuple[dict, dict]:
    """One method at one rate: per route {**warning_summary(units), "units": ...} and the thresholds."""
    units, thr = {}, {}
    for (g, s, c), (ref, te) in wi["scores"][m].items():
        flags, t = threshold_and_apply(ref, te, a)
        thr[threshold_key(m, g, s, c, rate)] = t
        units.setdefault(c, {})[(g, s)] = warning_unit(flags, wi["rows"][g], c, a)
    models = {c: {**warning_summary(u), "units": {f"{g}|s{s}": v for (g, s), v in u.items()}}
              for c, u in units.items()}
    return models, thr


def summary_only(block: dict) -> dict:
    return {k: v for k, v in block.items() if k != "units"}


def achieved_pooled(summary: dict):
    return summary["achieved_rate_exposure_weighted"]["nine_folds"]["rate_per_cell_hour"]


def achieved_mean(summary: dict):
    return summary["achieved_rate_equal_group_macro"]["nine_folds"]["macro"]


# ------------------------------------------------------------------ the cross-checks
def alarm_check(computed: dict, warnings: dict | None, image: dict | None, tol: float = CHECK_TOL) -> dict:
    """At 0.2 and 0.4: every DeadAhead and elapsed-time leaf (summary, units, thresholds) equals the warnings part and
    every image-baseline leaf equals the image-warnings part. ``computed`` is {(m, rate): (models, thresholds)}."""
    out, probs = {}, []
    for m in ALARM_METHODS:
        rec = image if m == IMAGE else warnings
        if rec is None:
            continue
        want_m, got_m, got_t = {}, {}, {}
        for rate in CHECK_RATES:
            models, thr = computed[(m, rate)]
            for c in CAUSES:
                want_m[f"{c}|{rate}"] = ((((rec or {}).get("models") or {}).get(m) or {}).get(c) or {}).get(rate)
                got_m[f"{c}|{rate}"] = models.get(c)
            got_t.update(thr)
        want_t = {k: v for k, v in ((rec or {}).get("thresholds") or {}).items()
                  if k.startswith(f"{m}|") and k.rsplit("|", 1)[-1] in CHECK_RATES}
        r = compare_tree(roundtrip({"models": want_m, "thresholds": want_t}),
                         roundtrip({"models": got_m, "thresholds": got_t}), tol)
        out[m] = {"max_abs_diff": r["max_abs_diff"], "n_leaves_compared": r["n_leaves"],
                  "n_problems": r["n_problems"], "problems": r["problems"][:10], "passes": r["passes"]}
        probs += [f"{m}: {p}" for p in r["problems"][:5]]
        if r["n_leaves"] == 0:
            out[m]["passes"] = False
            probs.append(f"{m}: no leaf compared")
    return {"tolerance": tol, "rates": list(CHECK_RATES), "methods": out, "problems": probs,
            "passes": all(v["passes"] for v in out.values()) and not probs}


def grid_check(sweep: dict, grid_full: dict, tol: float = GRID_TOL) -> dict:
    """The grid gives the same values as the sweep at every rate both hold (every summary leaf of both grid methods
    and routes)."""
    shared = [r for r in sweep[DEADAHEAD][LYTIC] if r in grid_full[DEADAHEAD][LYTIC]]
    a = {m: {c: {r: sweep[m][c][r] for r in shared} for c in sweep[m]} for m in GRID_METHODS}
    b = {m: {c: {r: grid_full[m][c][r] for r in shared} for c in grid_full[m]} for m in GRID_METHODS}
    t = compare_tree(roundtrip(a), roundtrip(b), tol)
    passes = bool(t["passes"] and len(shared) == SWEEP_STEPS and t["n_leaves"] > 0)
    return {"tolerance": tol, "shared_rates": shared, "n_shared": len(shared), "n_leaves_compared": t["n_leaves"],
            "max_abs_diff": t["max_abs_diff"], "problems": t["problems"][:10], "passes": passes}


# ------------------------------------------------------------------ the sweep, the grid, the match
def last_hour(summary: dict) -> dict:
    v = summary["sustained"][LAST_HOUR]
    return {"per_group": v["nine_folds"]["per_group"], "macro": v["nine_folds"]["macro"],
            "n_groups": v["nine_folds"]["n_groups"], "eligible_deaths_per_group": v["eligible_deaths_per_group"]}


def paired_last_hour(dead: dict, img: dict, what: str) -> dict:
    """DeadAhead minus the image baseline on the share warned in the last 60 min, per group (paired_both); the eligible
    deaths must be the same on both sides."""
    d, i = dead["sustained"][LAST_HOUR], img["sustained"][LAST_HOUR]
    if d["eligible_deaths_per_group"] != i["eligible_deaths_per_group"]:
        raise ValueError(f"{what}: the eligible deaths at X = {LAST_HOUR} differ between DeadAhead and the image "
                         "baseline")
    return paired_both(d["nine_folds"]["per_group"], i["nine_folds"]["per_group"])


def sweep_contrasts(sweep: dict) -> dict:
    """At every sweep rate and route: the eligible deaths equal for the three methods, and DeadAhead minus the image
    baseline at X = 60."""
    out = {}
    for c in sweep[DEADAHEAD]:
        for rate in sweep[DEADAHEAD][c]:
            e = [sweep[m][c][rate]["sustained"][LAST_HOUR]["eligible_deaths_per_group"] for m in ALARM_METHODS]
            if not e[0] == e[1] == e[2]:
                raise ValueError(f"{c} {rate}: the eligible deaths at X = {LAST_HOUR} differ between the methods")
            out[f"{DEADAHEAD} - {IMAGE}|{c}|{rate}|sustained {LAST_HOUR}"] = paired_last_hour(
                sweep[DEADAHEAD][c][rate], sweep[IMAGE][c][rate], f"{c} {rate}")
    return out


def _finite(v) -> bool:
    try:
        return v is not None and not isinstance(v, bool) and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def closest(candidates: dict, target) -> dict:
    """candidates {rate: (r, achieved pooled rate a)}: the rate whose a is closest to ``target`` (|a - target| in
    float64), ties to the lower a, then the lower r."""
    rows = [(abs(float(a) - float(target)), float(a), Fraction(r), name) for name, (r, a) in candidates.items()
            if _finite(a)]
    if not rows or not _finite(target):
        return {"not_computed": "no finite candidate or no finite target", "target": target}
    best = min(rows, key=lambda x: (x[0], x[1], x[2]))
    same_d = sorted((x for x in rows if x[0] == best[0] and x[3] != best[3]), key=lambda x: x[2])
    return {"rate": best[3], "r_per_cell_hour": float(best[2]), "r_exact": str(best[2]), "achieved_pooled": best[1],
            "target": float(target), "abs_difference": best[0],
            "tied_at_the_same_distance": [{"rate": x[3], "achieved_pooled": x[1]} for x in same_d],
            "n_candidates": len(rows), "n_non_finite_left_out": len(candidates) - len(rows),
            "rule": "the smallest |a - target| in float64; ties to the lower achieved rate, then the lower r"}


def grid_row(summary: dict, r: Fraction, a: Fraction) -> dict:
    """What is kept of one grid point: the matching inputs and the share warned in the last 60 min (per group, the
    nine-group macro and its group count, the eight-group macro); "r" itself is dropped on output."""
    v = summary["sustained"][LAST_HOUR]
    return {"r": r, "r_per_cell_hour": float(r), "alpha": str(a), "achieved_pooled": achieved_pooled(summary),
            "achieved_pooled_per_seed": summary["achieved_rate_exposure_weighted"]["nine_folds"]["per_seed"],
            "achieved_mean_over_groups": achieved_mean(summary),
            "share_warned_last_60_min": {"per_group": v["nine_folds"]["per_group"], "macro": v["nine_folds"]["macro"],
                                         "n_groups": v["nine_folds"]["n_groups"],
                                         "macro_eight_folds_without_PNT1A_bp":
                                             v["eight_folds_without_PNT1A_bp"]["macro"]}}


def rate_view(summary: dict, rate: str) -> dict:
    return {"rate": rate, "achieved_pooled": achieved_pooled(summary),
            "achieved_pooled_per_seed": summary["achieved_rate_exposure_weighted"]["nine_folds"]["per_seed"],
            "achieved_mean_over_groups": achieved_mean(summary),
            "share_warned_last_60_min": last_hour(summary)}


def matched_point(grid_full: dict, grid_rows: dict, sweep: dict, route: str, target: float, decides: str) -> dict:
    """DeadAhead's grid rate closest to ``target`` at ``route``; DeadAhead there, beside the image baseline at 0.2."""
    ch = closest({r: (v["r"], v["achieved_pooled"]) for r, v in grid_rows[DEADAHEAD][route].items()}, target)
    if "not_computed" in ch:
        return {"route": route, "match": ch, "decides": decides}
    dead = grid_full[DEADAHEAD][route][ch["rate"]]
    img = sweep[IMAGE][route][PRIMARY_RATE]
    return {"route": route, "decides": decides, "match": ch,
            "deadahead_at_the_matched_rate": {**rate_view(dead, ch["rate"]), "summary": dead},
            "dinov2_handcrafted_at_0.2": rate_view(img, PRIMARY_RATE),
            DIFFERENCE: paired_last_hour(dead, img, f"the {route} match")}


def matched_reverse(grid_full: dict, grid_rows: dict, sweep: dict, target: float) -> dict:
    """The image baseline's grid rate closest to DeadAhead's achieved pooled lytic rate at 0.2 (printed beside)."""
    ch = closest({r: (v["r"], v["achieved_pooled"]) for r, v in grid_rows[IMAGE][LYTIC].items()}, target)
    if "not_computed" in ch:
        return {"route": LYTIC, "match": ch, "decides": "nothing"}
    img = grid_full[IMAGE][LYTIC][ch["rate"]]
    dead = sweep[DEADAHEAD][LYTIC][PRIMARY_RATE]
    return {"route": LYTIC, "decides": "nothing (printed beside the matched comparison)", "match": ch,
            "dinov2_handcrafted_at_the_matched_rate": {**rate_view(img, ch["rate"]), "summary": img},
            "deadahead_at_0.2": rate_view(dead, PRIMARY_RATE),
            DIFFERENCE: paired_last_hour(dead, img, "the reverse match")}


def match_all(grid_full: dict, grid_rows: dict, sweep: dict, tg: dict) -> dict:
    """The lytic match, the one comparison of this part: it raises if it cannot be made. Beside it, deciding nothing:
    the apoptosis match and the reverse match, each named not_computed if it cannot be made."""
    lytic = matched_point(grid_full, grid_rows, sweep, LYTIC, tg["c_star"][LYTIC], "the matched comparison")
    if "not_computed" in lytic["match"]:
        raise ValueError(f"the lytic match cannot be made: {lytic['match']['not_computed']}"
                         f" (target {lytic['match'].get('target')})")
    return {"lytic": lytic,
            "apoptosis": matched_point(grid_full, grid_rows, sweep, APOPTOSIS, tg["c_star"][APOPTOSIS],
                                       "nothing (printed beside the lytic match)"),
            "reverse_lytic": matched_reverse(grid_full, grid_rows, sweep, tg["deadahead_lytic_0.2"])}


def match_targets(sweep: dict) -> dict:
    """c* per route (the image baseline's achieved pooled rate at 0.2) and DeadAhead's lytic value at 0.2 (the reverse
    match's target)."""
    return {"c_star": {c: achieved_pooled(sweep[IMAGE][c][PRIMARY_RATE]) for c in (LYTIC, APOPTOSIS)},
            "deadahead_lytic_0.2": achieved_pooled(sweep[DEADAHEAD][LYTIC][PRIMARY_RATE])}


# ------------------------------------------------------------------ the part
def _raise_on_failure(checks: dict) -> None:
    failed = [k for k, v in checks.items() if not v["passes"]]
    if failed:
        raise ValueError(f"the alarm-rate analysis failed a cross-check: {failed}: "
                         + "; ".join(p for k in failed for p in checks[k]["problems"][:2]))


def part_alarm_rate(bundle: dict, warnings: dict | None = None, image: dict | None = None) -> dict:
    """The sweep, the grid and the matched comparison on the warning rows; with ``warnings`` and ``image`` (those
    parts' results) the rates 0.2 and 0.4 are checked against them, leaf by leaf."""
    if "inner_predictions" not in bundle or "movies" not in bundle or \
            IMAGE not in bundle.get("inner_predictions_by_method", {}):
        raise ValueError("the bundle holds no inner predictions of DeadAhead and the image baseline or no movie "
                         "table: the alarm-rate analysis cannot be computed")
    index_predictions(bundle)
    index_image_predictions(bundle)
    rows, info = warning_population(bundle)
    wi = warning_inputs(bundle, rows)
    sw = sweep_rates()
    for rate, a in NOMINAL.items():
        if a != sw[rate][1]:
            raise ValueError(f"{rate}: alpha {sw[rate][1]} is not the warnings part's {a}")
    computed = {(m, rate): rate_pass(wi, m, rate, sw[rate][1]) for rate in CHECK_RATES for m in ALARM_METHODS}
    checks = {}
    if warnings is not None or image is not None:
        checks["rates_0.2_and_0.4_equal_the_warning_parts"] = alarm_check(computed, warnings, image)
        _raise_on_failure(checks)
    sweep = {m: {} for m in ALARM_METHODS}
    for rate, (r, a) in sw.items():
        for m in ALARM_METHODS:
            models = computed[(m, rate)][0] if (m, rate) in computed else rate_pass(wi, m, rate, a)[0]
            for c, blk in models.items():
                sweep[m].setdefault(c, {})[rate] = summary_only(blk)
    grid_full, grid_rows = {m: {} for m in GRID_METHODS}, {m: {} for m in GRID_METHODS}
    for rate, (r, a) in grid_rates().items():
        for m in GRID_METHODS:
            for c, blk in rate_pass(wi, m, rate, a)[0].items():
                s = summary_only(blk)
                grid_full[m].setdefault(c, {})[rate] = s
                grid_rows[m].setdefault(c, {})[rate] = grid_row(s, r, a)
    checks["grid_equals_the_sweep"] = grid_check(sweep, grid_full)
    _raise_on_failure(checks)
    con = sweep_contrasts(sweep)
    tg = match_targets(sweep)
    matched = match_all(grid_full, grid_rows, sweep, tg)
    grid_out = {m: {c: {k: {kk: vv for kk, vv in v.items() if kk != "r"} for k, v in d.items()}
                    for c, d in grid_rows[m].items()} for m in GRID_METHODS}
    return {"post_hoc": True,
            "rows": {"n": len(rows), "reference_rows": int(rows["negative"].sum()),
                     "cohort_removed": info["cohort_removed"]},
            "rates": {"unit": "alarms per event-free reference cell-hour (each reference row 0.5 cell-hour)",
                      "sweep": {k: {"r": str(r), "alpha": str(a)} for k, (r, a) in sw.items()},
                      "grid": f"r = k / {GRID_DEN}, k = 1..{GRID_STEPS} (alpha = k / {2 * GRID_DEN})"},
            "methods": {m: METHOD_LABELS[m] for m in ALARM_METHODS},
            "sweep": sweep, "contrasts_last_hour": con, "grid": grid_out,
            "match_targets": tg, "matched": matched, "checks": checks}


def _rate_value(name: str) -> float:
    return float(name.split("_", 1)[0])


def figure_csv(res: dict) -> str:
    """The values of the alarm-rate figure as CSV text: per route, method and target rate the achieved rates (pooled,
    mean over groups) and the percentage of deaths warned in the last 60 min (mean over groups); six decimals (the
    input of ``deadahead.report.figures``). Set sweep: the 16 targets; grid: the 160 targets; matched: the two sides
    of the lytic match."""
    head = ["route", "method", "set", "target_rate", "achieved_pooled", "achieved_mean_over_groups",
            "warned_last_60_min_pct"]
    lines = [",".join(head)]

    def row(c, m, kind, r, pooled, mean, share):
        lines.append(f"{c},{m},{kind},{r:.3f},{pooled:.6f},{mean:.6f},{100 * share:.6f}")

    for c in (LYTIC, APOPTOSIS):
        for m in ALARM_METHODS:
            for rate, s in res["sweep"][m][c].items():
                row(c, m, "sweep", _rate_value(rate), achieved_pooled(s), achieved_mean(s),
                    s["sustained"][LAST_HOUR]["nine_folds"]["macro"])
        for m in GRID_METHODS:
            for v in res["grid"][m][c].values():
                row(c, m, "grid", v["r_per_cell_hour"], v["achieved_pooled"], v["achieved_mean_over_groups"],
                    v["share_warned_last_60_min"]["macro"])
    mt = res["matched"]["lytic"]
    for m, side in ((DEADAHEAD, mt["deadahead_at_the_matched_rate"]), (IMAGE, mt["dinov2_handcrafted_at_0.2"])):
        row(LYTIC, m, "matched", _rate_value(side["rate"]), side["achieved_pooled"], side["achieved_mean_over_groups"],
            side["share_warned_last_60_min"]["macro"])
    return "\n".join(lines) + "\n"


def format_alarm_rate(res: dict) -> str:
    """The alarm-rate analysis as text: per route and target the achieved pooled rate and the percentage of deaths
    warned in the last 60 min of each method, then the matched comparison."""
    lines = ["Post hoc alarm-rate analysis: achieved pooled rate, deaths warned in the last 60 min (%, nine-group "
             "macro), training-only thresholds; image: DINOv2 + handcrafted; difference: DeadAhead minus image "
             "(percentage points), groups ahead of 9"]
    for c in (LYTIC, APOPTOSIS):
        lines += ["", c.lower(), f"{'target':>7}{'DeadAhead':>15}{'image':>15}{'elapsed':>15}{'difference':>12}"]
        for rate in res["sweep"][DEADAHEAD][c]:
            cells = [f"{achieved_pooled(res['sweep'][m][c][rate]):.2f}, "
                     f"{100 * res['sweep'][m][c][rate]['sustained'][LAST_HOUR]['nine_folds']['macro']:5.1f}"
                     for m in ALARM_METHODS]
            p = res["contrasts_last_hour"][f"{DEADAHEAD} - {IMAGE}|{c}|{rate}|sustained {LAST_HOUR}"]["nine_folds"]
            lines.append(f"{_rate_value(rate):>7.2f}{cells[0]:>15}{cells[1]:>15}{cells[2]:>15}"
                         f"{100 * p.get('macro_difference', float('nan')):>+8.1f}{p.get('x_ahead', 0):>2d}/{p['n']}")
    mt = res["matched"]["lytic"]
    d, i = mt["deadahead_at_the_matched_rate"], mt["dinov2_handcrafted_at_0.2"]
    p = mt[DIFFERENCE]["nine_folds"]
    lines += ["", ("Matched lytic comparison (the target was chosen post hoc on the held-out groups; each threshold set "
                   "on the training groups only)"),
              f"  DeadAhead at target {_rate_value(d['rate']):g}: achieved {d['achieved_pooled']:.3f} pooled, "
              f"{d['achieved_mean_over_groups']:.3f} mean over groups, warned "
              f"{100 * d['share_warned_last_60_min']['macro']:.1f}%",
              f"  image baseline at target {_rate_value(i['rate']):g}: achieved {i['achieved_pooled']:.3f} pooled, "
              f"{i['achieved_mean_over_groups']:.3f} mean over groups, warned "
              f"{100 * i['share_warned_last_60_min']['macro']:.1f}%",
              f"  DeadAhead minus the image baseline: {100 * p['macro_difference']:+.1f} points, ahead in "
              f"{p['x_ahead']} of {p['n']} groups, {outcome_text(p)}"]

    def beside(title: str, blk: dict, first: str, name_first: str, second: str, name_second: str) -> None:
        if "not_computed" in blk["match"]:
            lines.extend(["", f"{title}: {blk['match']['not_computed']}"])
            return
        a, b, q = blk[first], blk[second], blk[DIFFERENCE]["nine_folds"]
        lines.extend(["", f"{title} (decides nothing)",
                      f"  {name_first} at target {_rate_value(a['rate']):g}: achieved {a['achieved_pooled']:.3f} "
                      f"pooled, warned {100 * a['share_warned_last_60_min']['macro']:.1f}%",
                      f"  {name_second} at target {_rate_value(b['rate']):g}: achieved {b['achieved_pooled']:.3f} "
                      f"pooled, warned {100 * b['share_warned_last_60_min']['macro']:.1f}%",
                      f"  DeadAhead minus the image baseline: {100 * q['macro_difference']:+.1f} points, ahead in "
                      f"{q['x_ahead']} of {q['n']} groups, {outcome_text(q)}"])

    beside("Matched the other way round (lytic)", res["matched"]["reverse_lytic"],
           "dinov2_handcrafted_at_the_matched_rate", "image baseline", "deadahead_at_0.2", "DeadAhead")
    beside("Matched apoptosis", res["matched"]["apoptosis"], "deadahead_at_the_matched_rate", "DeadAhead",
           "dinov2_handcrafted_at_0.2", "image baseline")
    return "\n".join(lines)
