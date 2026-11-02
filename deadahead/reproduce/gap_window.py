"""Reproduce the post hoc window analysis: AUPRC of deaths g to H min ahead, with the forecasts whose cell died within
the first g min left out, at H = 120 min (g = 0, 30, 60, 90) and H = 240 min (g = 0, 30, 60, 90, 120).

Sets per cause k (T and C in minutes from the forecast): E = P_g or Q_g (left out); P = P_H and not P_g; Q = Q_H and
not Q_g (a negative); N = N_H; U = U_H, the rest (left out). AUPRC over P, Q and N. Score: F_k(t, H) as stored; elapsed
time uses its one-draw time score as it is. In the JSON only, for g > 0: the increase F_k(t, H) - F_k(t, g), where
F_k(t, g) is stored (g = 30, 60, 120; the forecast at 90 min is not stored). At H = 120 and g = 60 the increase is the
gap part's primary score.
"""

from __future__ import annotations

import math

import numpy as np

from deadahead.evaluation.aggregate import AHEAD_X, _finite, both_folds, paired_both
from deadahead.evaluation.alarms import NOMINAL, rows_id
from deadahead.evaluation.horizons import outcome_sets, precision_sensitivity
from deadahead.evaluation.rows import CAUSES, GROUPS, SEEDS, set_counts
from deadahead.evaluation.table1 import auprc_sets
from deadahead.reproduce.gap import (DEADAHEAD, IMAGE_SURVIVAL, METHOD_LABELS, SET_NAMES, _per_seed, determined,
                                     horizon_population, seeds_of)
from deadahead.reproduce.horizons import HORIZONS, cif_preds, prepare
from deadahead.reproduce.warning_times import TIME, TIME_SEED, time_for

WINDOWS = {120: (0, 30, 60, 90), 240: (0, 30, 60, 90, 120)}      # {H: gaps}
PRIMARY_H = 120
GAP_MODELS = (DEADAHEAD, IMAGE_SURVIVAL)                         # the two with stored cumulative incidences
GAP_METHODS = (*GAP_MODELS, TIME)
GAP_OTHERS = GAP_METHODS[1:]                                     # each contrast is DeadAhead minus one of these
CHECK_TOL = 1e-12                                                # the cross-checks with the gap and horizons parts
PRIMARY_LABEL = ("F_k(t, H) as stored, from the held-out predictions (elapsed time: its one-draw time score over the "
                 "whole population, as it is)")
SECONDARY_LABEL = ("the increase F_k(t, H) - F_k(t, g) where F_k(t, g) is stored (elapsed time keeps its time score); "
                   "AUPRC only, no contrasts; at H = 120 and g = 60 it is the gap part's primary score")
SET_TEXT = {"E": "a death of either route at T <= g (left out): P_g or Q_g",
            "P": "a death by route k with g < T <= H: P_H and not P_g",
            "Q": "a death by the other route with g < T <= H (a negative): Q_H and not Q_g",
            "N": "no death by H and followed through t + H (a negative): N_H",
            "U": "the rest, no death by H and follow-up ending before t + H (left out, never a negative): U_H"}


# ------------------------------------------------------------------ the sets and the scores
def window_sets(T, delta, Cm, cause: str, g: float, H: float) -> dict:
    """E, P, Q, N, U for route ``cause`` in the window (g, H] from (T, delta, C) in minutes, through outcome_sets at
    h = g and h = H."""
    if not 0 <= g < H:
        raise ValueError(f"a gap {g} outside [0, {H})")
    sg = outcome_sets(T, delta, Cm, cause, g)
    sh = outcome_sets(T, delta, Cm, cause, H)
    return {"E": sg["P"] | sg["Q"], "P": sh["P"] & ~sg["P"], "Q": sh["Q"] & ~sg["Q"], "N": sh["N"].copy(),
            "U": sh["U"].copy()}


def primary_score(m: str, preds: dict, cause: str, H: int):
    """F_k(t, H) as stored (elapsed time: its time score, the same at every horizon)."""
    return np.asarray(preds[H][cause], float)


def secondary_score(m: str, preds: dict, cause: str, g: int, H: int):
    """F_k(t, H) - F_k(t, g); elapsed time keeps its time score; None where F_k(t, g) is not stored."""
    late = np.asarray(preds[H][cause], float)
    if m == TIME:
        return late
    if g not in preds:
        return None
    return late - np.asarray(preds[g][cause], float)


def all_windows() -> list[tuple[int, int]]:
    return [(H, g) for H, gs in WINDOWS.items() for g in gs]


def context(bundle: dict, pop) -> dict:
    """The groups' rows and every method's scores at every stored horizon, read once.

    ``coll[m][g]`` is {"seeds": {s: {h: {cause: array over the group's rows}}}} (nan where a method has no row; elapsed
    time: the one-draw time score at every h) or {"failed": why} where the predictions cannot be read for the group."""
    hs = tuple(HORIZONS)
    rows = {g: pop[(pop["group"] == g).to_numpy()].reset_index(drop=True) for g in GROUPS}
    ts = np.asarray(time_for(pop), float)
    coll = {m: {} for m in GAP_METHODS}
    for g in GROUPS:
        pg = rows[g]
        for m in GAP_MODELS:
            try:
                coll[m][g] = {"seeds": {s: cif_preds(bundle, m, g, s, pg, hs) for s in SEEDS}}
            except ValueError as e:
                coll[m][g] = {"failed": f"a stored file cannot be read for the group: {e}"}
        t = ts[(pop["group"] == g).to_numpy()]
        coll[TIME][g] = {"seeds": {TIME_SEED: {h: {c: t for c in CAUSES} for h in hs}}}
    return {"pop": pop, "rows": rows, "coll": coll}


def window_set_table(ctx: dict, g: int, H: int) -> dict:
    return {c: {grp: window_sets(pg["T_min"], pg["delta"], pg["C_min"], c, g, H) for grp, pg in ctx["rows"].items()}
            for c in CAUSES}


def method_status(ctx: dict, sets: dict, H: int) -> tuple[dict, dict]:
    """One window: per method scored ({g: the seeds' predictions}) or not computed (why: the whole method for this
    window, both routes, every group, every seed)."""
    scored, not_computed = {}, {}
    for m in GAP_METHODS:
        why, have = None, {}
        for g in GROUPS:
            e = ctx["coll"][m][g]
            if "failed" in e:
                why = why or f"NOT COMPUTED: {g}: {e['failed']}"
                continue
            if set(e["seeds"]) != set(seeds_of(m)):
                why = why or (f"NOT COMPUTED: {g}: seeds {sorted(e['seeds'])}, not {list(seeds_of(m))} (a missing "
                              "unit)")
                continue
            for s, preds in e["seeds"].items():
                for c in CAUSES:
                    det = determined(sets[c][g])
                    bad = ~np.isfinite(primary_score(m, preds, c, H)[det])
                    if bad.any() and why is None:
                        why = (f"NOT COMPUTED: {g} seed {s} route {c}: {int(bad.sum())} determined row(s) (P, Q or "
                               "N) without a finite score")
            have[g] = e["seeds"]
        if why:
            not_computed[m] = why
        else:
            scored[m] = have
    return scored, not_computed


def secondary_status(scored: dict, sets: dict, g: int, H: int) -> dict:
    """{method: why} where the increase is not finite on a determined row (the increase only, the whole method)."""
    out = {}
    for m, have in scored.items():
        if m == TIME:
            continue
        for grp, seeds in have.items():
            for s, preds in seeds.items():
                for c in CAUSES:
                    d = secondary_score(m, preds, c, g, H)
                    if d is None:
                        continue
                    bad = ~np.isfinite(d[determined(sets[c][grp])])
                    if bad.any() and m not in out:
                        out[m] = (f"NOT COMPUTED: {grp} seed {s} route {c}: {int(bad.sum())} determined row(s) without "
                                  f"a finite F_k(t, {H}) - F_k(t, {g})")
    return out


# ------------------------------------------------------------------ one window
def window_block(ctx: dict, g: int, H: int) -> dict:
    """One window (g, H]: per route the counts, chance, each method's primary AUPRC per seed and group with the macros,
    the increase, and DeadAhead minus each other method."""
    rows = ctx["rows"]
    sets = window_set_table(ctx, g, H)
    scored, not_computed = method_status(ctx, sets, H)
    sec_nc = secondary_status(scored, sets, g, H) if g > 0 else {}
    out = {}
    for c in CAUSES:
        counts = {grp: {**set_counts(sets[c][grp]), "rows": len(rows[grp])} for grp in GROUPS}
        prev = {grp: precision_sensitivity(np.zeros(len(rows[grp]), bool), sets[c][grp])["prevalence"]
                for grp in GROUPS}
        chance = both_folds({grp: {TIME_SEED: v} for grp, v in prev.items()})
        fin = {grp: v for grp, v in prev.items() if _finite(v)}
        blk = {"counts": {"per_group": counts, "total": {k: int(sum(v[k] for v in counts.values()))
                                                        for k in (*SET_NAMES, "rows")},
                          "note": "the same rows for every method; E and U are left out, Q and N are negatives"},
               "chance": {"rule": "each group's |P| / (|P| + |Q| + |N|), the macro over the 9 groups", **chance,
                          "per_group_range": {"min": min(fin.values()) if fin else None,
                                              "max": max(fin.values()) if fin else None,
                                              "min_group": min(fin, key=fin.get) if fin else None,
                                              "max_group": max(fin, key=fin.get) if fin else None}},
               "methods": {}, "contrasts": {}}
        for m in GAP_METHODS:
            e = {"label": METHOD_LABELS[m], "score": PRIMARY_LABEL, "not_computed": not_computed.get(m)}
            if m in not_computed:
                e.update(primary=None, secondary=None)
                blk["methods"][m] = e
                continue
            have = scored[m]
            prim = {grp: {s: auprc_sets(primary_score(m, p, c, H), sets[c][grp]) for s, p in seeds.items()}
                    for grp, seeds in have.items()}
            pb = both_folds(prim)
            keep = pb["nine_folds"]["groups_contributing"]
            e["primary"] = {"per_seed": _per_seed(prim), **pb, "chance_over_the_same_groups": {
                "macro": float(np.mean([prev[grp] for grp in keep])) if keep else float("nan"), "groups": keep,
                "n_groups": len(keep)}}
            if g == 0:
                e["secondary"] = {"not_applicable": "g = 0: the increase is defined for g > 0"}
            elif m in sec_nc:
                e["secondary"] = {"label": SECONDARY_LABEL, "not_computed": sec_nc[m]}
            elif m != TIME and g not in HORIZONS:
                e["secondary"] = {"label": SECONDARY_LABEL,
                                  "not_stored": f"F_k(t, {g}) is not stored (stored horizons {list(HORIZONS)})"}
            else:
                sec = {grp: {s: auprc_sets(secondary_score(m, p, c, g, H), sets[c][grp]) for s, p in seeds.items()}
                       for grp, seeds in have.items()}
                e["secondary"] = {"label": SECONDARY_LABEL, "per_seed": _per_seed(sec), **both_folds(sec)}
                if m == TIME:
                    e["secondary"]["note"] = "elapsed time keeps its time score: the values are its primary's"
            blk["methods"][m] = e
        x = blk["methods"][DEADAHEAD]
        for m in GAP_OTHERS:
            y = blk["methods"][m]
            key = f"{DEADAHEAD} - {m}"
            if x["not_computed"] or y["not_computed"]:
                blk["contrasts"][key] = {"not_computed": x["not_computed"] or y["not_computed"]}
                continue
            k = paired_both(x["primary"]["nine_folds"]["per_group"], y["primary"]["nine_folds"]["per_group"])
            k["deadahead_called_ahead"] = k["nine_folds"].get("outcome") == AHEAD_X
            blk["contrasts"][key] = k
        out[c] = blk
    return {"window_min": {"gap": g, "horizon": H}, "routes": out}


def gap_summary(windows: dict) -> dict:
    """A compact table of the windows: per H, g and route the chance, each method's primary and increase macros with
    their group counts, and each contrast's macro difference, groups ahead and outcome."""
    out = {}
    for (H, g), w in windows.items():
        for c, blk in w["routes"].items():
            row = {"chance": blk["chance"]["nine_folds"]["macro"], "methods": {}, "contrasts": {}}
            for m, e in blk["methods"].items():
                p, s = e.get("primary"), e.get("secondary") or {}
                row["methods"][m] = {
                    "primary_macro": p["nine_folds"]["macro"] if p else None,
                    "n_groups": p["nine_folds"]["n_groups"] if p else None,
                    "chance_over_the_same_groups": p["chance_over_the_same_groups"]["macro"] if p else None,
                    "secondary_macro": s["nine_folds"]["macro"] if "nine_folds" in s else None,
                    "not_computed": e.get("not_computed")}
            for k, v in blk["contrasts"].items():
                nf = v.get("nine_folds") or {}
                row["contrasts"][k] = {"macro_difference": nf.get("macro_difference"), "x_ahead": nf.get("x_ahead"),
                                       "n": nf.get("n"), "outcome": nf.get("outcome") or v.get("not_computed"),
                                       "deadahead_called_ahead": bool(v.get("deadahead_called_ahead"))}
            out.setdefault(str(H), {}).setdefault(str(g), {})[c] = row
    return out


# ------------------------------------------------------------------ the cross-checks
def partition(pop, windows) -> list[str]:
    """Every row in one of the 9 groups, and E, P, Q, N, U disjoint and exhaustive in every group at both routes in
    every window given; the problems found."""
    probs = []
    outside = sorted(set(pop["group"].astype(str)) - set(GROUPS))
    if outside:
        probs.append(f"rows of groups outside the 9: {outside}")
    groups = {grp: pop[(pop["group"] == grp).to_numpy()] for grp in GROUPS}
    for H, g in windows:
        for c in CAUSES:
            for grp, pg in groups.items():
                sets = window_sets(pg["T_min"], pg["delta"], pg["C_min"], c, g, H)
                k = np.stack([np.asarray(sets[x], bool) for x in SET_NAMES]).astype(int).sum(axis=0)
                if not (k == 1).all():
                    probs.append(f"H {H} g {g} {c} {grp}: {int((k != 1).sum())} row(s) not in exactly one of E, P, Q, "
                                 "N, U")
    return probs


def g0_problems(ctx: dict, H: int) -> list[str]:
    """At g = 0, E is empty and P, Q, N, U are outcome_sets' at h = H, in every group and route."""
    probs = []
    for c in CAUSES:
        for grp, pg in ctx["rows"].items():
            w = window_sets(pg["T_min"], pg["delta"], pg["C_min"], c, 0, H)
            h = outcome_sets(pg["T_min"], pg["delta"], pg["C_min"], c, H)
            if w["E"].any():
                probs.append(f"H {H} g 0 {c} {grp}: {int(w['E'].sum())} row(s) in E (a death at T <= 0)")
            bad = [k for k in ("P", "Q", "N", "U") if not np.array_equal(w[k], h[k])]
            if bad:
                probs.append(f"H {H} g 0 {c} {grp}: {bad} differ from outcome_sets at h = {H}")
    return probs


def _diff(a, b) -> float:
    """|a - b|; 0 when both are nan, inf when one is not a number."""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return math.inf
    if math.isnan(a) and math.isnan(b):
        return 0.0
    d = abs(a - b)
    return math.inf if math.isnan(d) else d


def compare_block(ours: dict, want: dict | None, tol: float = CHECK_TOL) -> list[str]:
    """``ours`` (both_folds) against a block of another part: the nine-group macro and every per-group value to
    ``tol``, the same groups; the problems found."""
    if not isinstance(want, dict) or not isinstance(want.get("nine_folds"), dict):
        return ["missing in the other part"]
    o9, w9 = ours["nine_folds"], want["nine_folds"]
    probs = []
    dm = _diff(o9.get("macro"), w9.get("macro"))
    if not dm <= tol:
        probs.append(f"the nine-group macro differs by {dm:.3g} > {tol:g}")
    op, wp = o9.get("per_group") or {}, w9.get("per_group") or {}
    if set(op) != set(wp):
        probs.append(f"groups {sorted(op)} here, {sorted(wp)} in the other part")
    dg = max((_diff(op[g], wp[g]) for g in set(op) & set(wp)), default=0.0)
    if not dg <= tol:
        probs.append(f"a per-group value differs by {dg:.3g} > {tol:g}")
    return probs


def horizon_blocks(horizons: dict, m: str, cause: str, h: int) -> list[tuple[str, dict | None]]:
    """The horizons part's blocks for (m, cause, h), each holding "auprc" and "prevalence"."""
    if m in (DEADAHEAD, TIME):
        rates = ((horizons.get("models") or {}).get(m) or {}).get(cause, {}).get(str(h))
        if not rates:
            return [(f"models.{m}.{cause}.{h}", None)]
        return [(f"models.{m}.{cause}.{h}.{r}", rates.get(r)) for r in NOMINAL]
    blk = ((horizons.get("cif_models_auprc") or {}).get(m) or {}).get(cause, {}).get(str(h))
    return [(f"cif_models_auprc.{m}.{cause}.{h}", blk)]


def check_g0(windows: dict, horizons: dict, tol: float = CHECK_TOL) -> dict:
    """At g = 0 and H = 120, 240 each method's AUPRC and prevalence equal the horizons part's."""
    probs, n = [], 0
    for H in WINDOWS:
        routes = windows[(H, 0)]["routes"]
        for m in GAP_METHODS:
            for c in CAUSES:
                blk = routes[c]
                e = blk["methods"][m]
                ours = {"auprc": e.get("primary"),
                        "prevalence": {k: blk["chance"][k] for k in ("nine_folds", "eight_folds_without_PNT1A_bp")}}
                for path, want in horizon_blocks(horizons, m, c, H):
                    for k in ("auprc", "prevalence"):
                        n += 1
                        if ours[k] is None:
                            probs.append(f"{path}.{k}: {e['not_computed']} here")
                            continue
                        w = want.get(k) if isinstance(want, dict) else None
                        probs += [f"{path}.{k}: {p}" for p in compare_block(ours[k], w, tol)]
    return {"n_checks": n, "tolerance": tol, "problems": probs[:10], "n_problems": len(probs), "passes": not probs}


def check_g60(block: dict, gap: dict, tol: float = CHECK_TOL) -> dict:
    """At H = 120 and g = 60 the increase of DeadAhead and the image survival baseline, elapsed time's AUPRC and the
    counts of E, P, Q, N and U equal the gap part's."""
    probs, n = [], 0
    for c in CAUSES:
        mine = block["routes"][c]
        theirs = gap["routes"][c]
        for m in GAP_METHODS:
            e = mine["methods"][m]
            ours = e.get("primary") if m == TIME else e.get("secondary")
            n += 1
            if not isinstance(ours, dict) or "nine_folds" not in ours:
                probs.append(f"{c} {m}: not scored here")
                continue
            want = (theirs["methods"].get(m) or {}).get("primary")
            probs += [f"{c} {m}: {p}" for p in compare_block(ours, want, tol)]
        for grp, v in mine["counts"]["per_group"].items():
            w = theirs["counts"]["per_group"].get(grp) or {}
            probs += [f"{c} {grp} |{k}|: {v[k]} here, {w.get(k)} in the gap part" for k in SET_NAMES if w.get(k) != v[k]]
    return {"n_checks": n, "tolerance": tol, "problems": probs[:10], "n_problems": len(probs), "passes": not probs}


# ------------------------------------------------------------------ the part
def part_gap_window(bundle: dict, horizons: dict | None = None, gap: dict | None = None) -> dict:
    """The window analysis on the survival population of the evaluation cohort; with ``horizons`` and ``gap`` (those
    parts' results) its g = 0 and g = 60 windows are checked against them."""
    prepare(bundle)
    pop, removed = horizon_population(bundle)
    probs = partition(pop, all_windows())
    if probs:
        raise ValueError("the outcome sets do not partition the rows: " + "; ".join(probs[:3]))
    ctx = context(bundle, pop)
    probs = [p for H in WINDOWS for p in g0_problems(ctx, H)]
    if probs:
        raise ValueError("the sets at g = 0 are not those of h = H: " + "; ".join(probs[:3]))
    ordered = {w: window_block(ctx, w[1], w[0]) for w in all_windows()}
    checks = {"partition": {"n_rows": len(pop), "windows_checked": [f"{H}|{g}" for H, g in all_windows()],
                            "passes": True}}
    if horizons is not None:
        checks["g0_equals_horizons_part"] = check_g0(ordered, horizons)
    if gap is not None:
        checks["g60_equals_gap_part"] = check_g60(ordered[(120, 60)], gap)
    failed = [k for k, v in checks.items() if not v["passes"]]
    if failed:
        raise ValueError(f"the window analysis disagrees with another part: {failed}: "
                         + "; ".join(p for k in failed for p in checks[k]["problems"][:2]))
    return {"post_hoc": True,
            "population": {"n_rows": len(pop), "rows_id": rows_id(pop), "cohort_removed": removed},
            "sets": SET_TEXT, "scores": {"primary": PRIMARY_LABEL, "secondary": SECONDARY_LABEL},
            "methods": {m: METHOD_LABELS[m] for m in GAP_METHODS},
            "windows_min": {"primary": {"H": PRIMARY_H, "g": list(WINDOWS[PRIMARY_H])},
                            "secondary": {"H": 240, "g": list(WINDOWS[240])}},
            "windows": {str(H): {str(g): ordered[(HH, g)] for (HH, g) in ordered if HH == H} for H in WINDOWS},
            "summary": gap_summary(ordered), "checks": checks}


def _cell(v, fmt: str = ".6f") -> str:
    return "" if v is None else format(v, fmt)


def figure_csv(res: dict) -> str:
    """The values of the window figure as CSV text: per window, route and method the macro AUPRC, chance and DeadAhead
    minus the method (macro difference, groups ahead); six decimals (the input of ``deadahead.report.figures``)."""
    head = ["horizon_min", "gap_min", "route", "method", "auprc", "chance", "deadahead_minus_method", "groups_ahead",
            "n_groups", "meets_decision_rule"]
    lines = [",".join(head)]
    for H, by_g in res["summary"].items():
        for g in sorted(by_g, key=int):
            for c in ("LYTIC", "APOPTOSIS"):
                s = by_g[g][c]
                for m in GAP_METHODS:
                    e, k = s["methods"][m], s["contrasts"].get(f"{DEADAHEAD} - {m}") or {}
                    lines.append(",".join([H, g, c, m, _cell(e["primary_macro"]), _cell(s["chance"]),
                                           _cell(k.get("macro_difference")), _cell(k.get("x_ahead"), "d"),
                                           _cell(e["n_groups"], "d"), str(k.get("deadahead_called_ahead", ""))]))
    return "\n".join(lines) + "\n"


def format_gap_window(res: dict) -> str:
    """The window analysis as text: macro AUPRC per window, route and method, and DeadAhead's leads."""
    lines = [("Post hoc window analysis: AUPRC of deaths g to H min ahead, forecasts whose cell died within g min left "
              "out (nine-group macro)")]
    for H, by_g in res["summary"].items():
        for c in ("LYTIC", "APOPTOSIS"):
            lines += ["", f"H = {H} min, {c.lower()}",
                      f"{'g (min)':>8}{'chance':>9}{'DeadAhead':>11}{'image':>9}{'elapsed':>9}"
                      f"{'minus image':>14}{'minus elapsed':>16}"]
            for g in sorted(by_g, key=int):
                s = by_g[g][c]
                v = [_cell(s["methods"][m]["primary_macro"], ".3f") for m in GAP_METHODS]
                cells = []
                for m in GAP_OTHERS:
                    k = s["contrasts"][f"{DEADAHEAD} - {m}"]
                    mark = "*" if k["deadahead_called_ahead"] else " "
                    cells.append(f"{_cell(k['macro_difference'], '+.3f')} {k['x_ahead']}/{k['n']}{mark}")
                lines.append(f"{g:>8}{s['chance']:>9.3f}{v[0]:>11}{v[1]:>9}{v[2]:>9}{cells[0]:>14}{cells[1]:>16}")
    lines += ["", "image: DINOv2 + handcrafted under the same survival head; * meets the decision rule"]
    return "\n".join(lines)
