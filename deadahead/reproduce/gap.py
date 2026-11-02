"""Reproduce the post hoc gap analysis: deaths 60 to 120 min ahead, with the forecasts whose cell died within the first
60 min left out.

Sets per cause k (T and C in minutes from the forecast): E = P60 or Q60 (left out); P = P120 and not P60;
Q = Q120 and not Q60 (a negative); N = N120; U = the rest (follow-up ends before t + 120 with no death; left out).
AUPRC over P, Q and N. Score: dF_k(t) = F_k(t, 120) - F_k(t, 60) from the stored predictions of the four survival
methods; elapsed time uses its one-draw time score as it is. Printed beside it: F_k(t, 120) and the forecast given
survival to t + 60, dF_k(t) / [1 - F_APOPTOSIS(t, 60) - F_LYTIC(t, 60)].
"""

from __future__ import annotations

import numpy as np

from deadahead.evaluation.aggregate import _finite, both_folds, outcome_text, paired_both
from deadahead.evaluation.alarms import oracle_outcomes, rows_id
from deadahead.evaluation.horizons import outcome_sets, precision_sensitivity
from deadahead.evaluation.rows import CAUSES, GROUPS, KEY, SEEDS, apply_cohort, key_index, set_counts
from deadahead.evaluation.table1 import auprc_sets
from deadahead.reproduce.warning_times import TIME, TIME_SEED, time_for

H_EARLY, H_LATE = 60, 120
SET_NAMES = ("E", "P", "Q", "N", "U")
DEADAHEAD, IMAGE_SURVIVAL, HANDCRAFTED_SURVIVAL, DEEPHIT = ("deadahead", "dinov2_handcrafted_survival",
                                                            "handcrafted_survival", "deephit")
SURVIVAL_METHODS = (DEADAHEAD, IMAGE_SURVIVAL, HANDCRAFTED_SURVIVAL, DEEPHIT)   # stored cumulative incidences
METHOD_ORDER = (*SURVIVAL_METHODS, TIME)
OTHERS = METHOD_ORDER[1:]                                                       # DeadAhead minus each of these
ROUTE_SUFFIX = {"LYTIC": "L", "APOPTOSIS": "A"}
SLOTS = {"x": DEADAHEAD, "z": IMAGE_SURVIVAL, "w": TIME}                       # y is chance
METHOD_LABELS = {
    DEADAHEAD: "DeadAhead: the survival head on the fine-tuned video features and the handcrafted features",
    IMAGE_SURVIVAL: "DINOv2 image features + handcrafted features under the same survival head",
    HANDCRAFTED_SURVIVAL: "the handcrafted features under the same survival head",
    DEEPHIT: "DeepHit on the DeadAhead inputs",
    TIME: "elapsed time in the movie: the one-draw time score over the whole population, used as it is",
}
PRIMARY_LABEL = "dF_k(t) = F_k(t, 120) - F_k(t, 60) from the stored predictions (elapsed time: its time score)"
F120_LABEL = "F_k(t, 120) as stored (elapsed time: its time score); printed beside the primary score"
GIVEN_LABEL = ("the forecast given survival to t + 60: dF_k(t) / [1 - F_APOPTOSIS(t, 60) - F_LYTIC(t, 60)]; printed "
               "beside the primary score")
TIME_NO_GIVEN = "not applicable: elapsed time has no cumulative incidence"
SET_TEXT = {"E": "P60 or Q60: the cell died within 60 min (left out)",
            "P": "a death of the cause 60 to 120 min ahead",
            "Q": "a death of the other cause 60 to 120 min ahead (a negative)",
            "N": "no death through t + 120 and followed through it (a negative)",
            "U": "follow-up ends before t + 120 with no death (left out)"}


# ------------------------------------------------------------------ the population and the stored predictions
def horizon_population(bundle: dict):
    """The survival population on the evaluation cohort: every landmark row with its complete-movie outcome
    (T, delta, C in minutes), the untyped-death cells removed, sorted by key; and the counts removed."""
    o = oracle_outcomes(bundle["landmarks"], bundle["events"], bundle["movies"])
    pop, removed = apply_cohort(o, bundle["untyped"])
    if pop["untyped_track"].any():
        raise ValueError("an untyped-death cell is still in the population after the cohort")
    return pop.sort_values(KEY).reset_index(drop=True), removed


def index_cumulative_incidence(bundle: dict) -> None:
    """Split the stored cumulative incidences once by (method, group, seed)."""
    if "ci_by_unit" not in bundle:
        q = bundle["cumulative_incidence"]
        bundle["ci_by_unit"] = {(m, g, int(s)): d for (m, g, s), d in q.groupby(["method", "group", "seed"])}


def dh_not_run_groups(bundle: dict) -> set:
    """The groups without stored DeepHit predictions (none in the released bundle)."""
    index_cumulative_incidence(bundle)
    return set(GROUPS) - {g for (m, g, s) in bundle["ci_by_unit"] if m == DEEPHIT}


def cif_preds(bundle: dict, model: str, g: str, s: int, pg, horizons) -> dict:
    """{h: {cause: array over pg}} of a method's stored F_k(t, h) at (g, s); nan where it has no row."""
    z = bundle["ci_by_unit"].get((model, g, int(s)))
    if z is None:
        raise ValueError(f"{model} {g} s{s}: no stored predictions")
    pos = key_index(z).get_indexer(key_index(pg))
    out = {}
    for h in horizons:
        out[h] = {}
        for c in CAUSES:
            p = z[f"F_{c}_{h}"].to_numpy(float)
            out[h][c] = np.where(pos >= 0, p[np.maximum(pos, 0)], np.nan)
    return out


# ------------------------------------------------------------------ the sets and the scores
def gap_sets(T, delta, Cm, cause: str) -> dict:
    """E, P, Q, N, U for cause ``cause`` from (T, delta, C) in minutes, through outcome_sets at h = 60 and 120."""
    s60 = outcome_sets(T, delta, Cm, cause, H_EARLY)
    s120 = outcome_sets(T, delta, Cm, cause, H_LATE)
    return {"E": s60["P"] | s60["Q"], "P": s120["P"] & ~s60["P"], "Q": s120["Q"] & ~s60["Q"],
            "N": s120["N"].copy(), "U": s120["U"].copy()}


def determined(sets: dict):
    return sets["P"] | sets["Q"] | sets["N"]


def primary_score(m: str, preds: dict, cause: str):
    """dF_k = F_k(120) - F_k(60); elapsed time's score as it is (the same at every horizon, never a difference)."""
    late = np.asarray(preds[H_LATE][cause], float)
    if m == TIME:
        return late
    return late - np.asarray(preds[H_EARLY][cause], float)


def f120_score(m: str, preds: dict, cause: str):
    """F_k(120) as stored (elapsed time: its time score)."""
    return np.asarray(preds[H_LATE][cause], float)


def given_survival_score(m: str, preds: dict, cause: str):
    """dF_k / [1 - F_APOPTOSIS(60) - F_LYTIC(60)]; nan where that survival is not positive; None for elapsed time
    (no cumulative incidence)."""
    if m == TIME:
        return None
    surv60 = 1.0 - np.asarray(preds[H_EARLY]["APOPTOSIS"], float) - np.asarray(preds[H_EARLY]["LYTIC"], float)
    d = primary_score(m, preds, cause)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(surv60 > 0, d / np.where(surv60 > 0, surv60, 1.0), np.nan)


def seeds_of(m: str) -> tuple:
    return (TIME_SEED,) if m == TIME else tuple(SEEDS)


def context(bundle: dict, pop) -> dict:
    """The groups' rows and every method's scores at h = 60 and 120, read once.

    ``coll[m][g]`` is {"seeds": {s: {h: {cause: array over the group's rows}}}}, {"not_run": why} for a group
    without DeepHit predictions, or {"failed": why} where a method's predictions cannot be read for the group."""
    rows = {g: pop[(pop["group"] == g).to_numpy()].reset_index(drop=True) for g in GROUPS}
    ts = np.asarray(time_for(pop), float)
    not_run = dh_not_run_groups(bundle)
    coll = {m: {} for m in METHOD_ORDER}
    for g in GROUPS:
        pg = rows[g]
        for m in SURVIVAL_METHODS:
            if m == DEEPHIT and g in not_run:
                coll[m][g] = {"not_run": "NOT RUN: a NOT RUN DeepHit unit removes every seed of its group"}
                continue
            try:
                coll[m][g] = {"seeds": {s: cif_preds(bundle, m, g, s, pg, (H_EARLY, H_LATE)) for s in SEEDS}}
            except ValueError as e:
                coll[m][g] = {"failed": f"a stored file cannot be read for the group: {e}"}
        t = ts[(pop["group"] == g).to_numpy()]
        coll[TIME][g] = {"seeds": {TIME_SEED: {H_EARLY: {c: t for c in CAUSES}, H_LATE: {c: t for c in CAUSES}}}}
    return {"pop": pop, "rows": rows, "coll": coll}


# ------------------------------------------------------------------ the gap analysis
def method_status(ctx: dict, sets: dict) -> tuple[dict, dict, dict]:
    """Per method: scored ({g: the seeds' predictions}), not computed (why: the whole method, both causes, every
    group, every seed) or groups not run (named)."""
    scored, not_computed, not_run = {}, {}, {}
    for m in METHOD_ORDER:
        why, have = None, {}
        for g in GROUPS:
            e = ctx["coll"][m][g]
            if "not_run" in e:
                not_run.setdefault(m, {})[g] = e["not_run"]
                continue
            if "failed" in e:
                why = why or f"NOT COMPUTED: {g}: {e['failed']}"
                continue
            if set(e["seeds"]) != set(seeds_of(m)):
                why = why or f"NOT COMPUTED: {g}: seeds {sorted(e['seeds'])}, not {list(seeds_of(m))} (a missing unit)"
                continue
            for s, preds in e["seeds"].items():
                for c in CAUSES:
                    det = determined(sets[c][g])
                    bad = ~np.isfinite(primary_score(m, preds, c)[det])
                    if bad.any() and why is None:
                        why = (f"NOT COMPUTED: {g} seed {s} route {c}: {int(bad.sum())} determined row(s) (P, Q or "
                               "N) without a finite score")
            have[g] = e["seeds"]
        if why:
            not_computed[m] = why
        else:
            scored[m] = have
    return scored, not_computed, not_run


def given_status(scored: dict, sets: dict) -> dict:
    """{method: why} where the forecast given survival to t + 60 is not finite on a determined row."""
    out = {}
    for m, have in scored.items():
        if m == TIME:
            continue
        for g, seeds in have.items():
            for s, preds in seeds.items():
                for c in CAUSES:
                    bad = ~np.isfinite(given_survival_score(m, preds, c)[determined(sets[c][g])])
                    if bad.any() and m not in out:
                        out[m] = (f"NOT COMPUTED: {g} seed {s} route {c}: {int(bad.sum())} determined row(s) where "
                                  "1 - F_APOPTOSIS(t, 60) - F_LYTIC(t, 60) is not positive")
    return out


def _per_seed(d: dict) -> dict:
    return {g: {str(s): v for s, v in x.items()} for g, x in d.items()}


def gap_block(ctx: dict) -> dict:
    """Per cause: the set counts, chance, each method's AUPRC (primary score and the two printed scores) per seed and
    group with the macros, and DeadAhead minus each other method."""
    rows = ctx["rows"]
    sets = {c: {g: gap_sets(pg["T_min"], pg["delta"], pg["C_min"], c) for g, pg in rows.items()} for c in CAUSES}
    scored, not_computed, not_run = method_status(ctx, sets)
    given_nc = given_status(scored, sets)
    out = {}
    for c in CAUSES:
        counts = {g: {**set_counts(sets[c][g]), "rows": len(rows[g])} for g in GROUPS}
        prev = {g: precision_sensitivity(np.zeros(len(rows[g]), bool), sets[c][g])["prevalence"] for g in GROUPS}
        chance = both_folds({g: {TIME_SEED: v} for g, v in prev.items()})
        fin = {g: v for g, v in prev.items() if _finite(v)}
        blk = {"counts": {"per_group": counts, "total": {k: int(sum(v[k] for v in counts.values()))
                                                        for k in (*SET_NAMES, "rows")},
                          "note": "the same rows for every method; E and U are left out, Q and N are negatives"},
               "chance": {"rule": "each group's |P| / (|P| + |Q| + |N|), the macro over the 9 groups", **chance,
                          "per_group_range": {"min": min(fin.values()) if fin else None,
                                              "max": max(fin.values()) if fin else None,
                                              "min_group": min(fin, key=fin.get) if fin else None,
                                              "max_group": max(fin, key=fin.get) if fin else None}},
               "methods": {}, "contrasts": {}}
        for m in METHOD_ORDER:
            e = {"label": METHOD_LABELS[m], "score": PRIMARY_LABEL, "not_run": dict(not_run.get(m) or {}),
                 "not_computed": not_computed.get(m)}
            if m in not_computed:
                e.update(primary=None, prints=None)
                blk["methods"][m] = e
                continue
            have = scored[m]
            prim = {g: {s: auprc_sets(primary_score(m, p, c), sets[c][g]) for s, p in seeds.items()}
                    for g, seeds in have.items()}
            f120 = {g: {s: auprc_sets(f120_score(m, p, c), sets[c][g]) for s, p in seeds.items()}
                    for g, seeds in have.items()}
            pb = both_folds(prim)
            keep = pb["nine_folds"]["groups_contributing"]
            e["primary"] = {"per_seed": _per_seed(prim), **pb, "chance_over_the_same_groups": {
                "macro": float(np.mean([prev[g] for g in keep])) if keep else float("nan"), "groups": keep,
                "n_groups": len(keep)}}
            prints = {"f120": {"label": F120_LABEL, "per_seed": _per_seed(f120), **both_folds(f120)}}
            if m == TIME:
                prints["given_survival_to_t_plus_60"] = {"label": GIVEN_LABEL, "not_applicable": TIME_NO_GIVEN}
            elif m in given_nc:
                prints["given_survival_to_t_plus_60"] = {"label": GIVEN_LABEL, "not_computed": given_nc[m]}
            else:
                gv = {g: {s: auprc_sets(given_survival_score(m, p, c), sets[c][g]) for s, p in seeds.items()}
                      for g, seeds in have.items()}
                prints["given_survival_to_t_plus_60"] = {"label": GIVEN_LABEL, "per_seed": _per_seed(gv),
                                                         **both_folds(gv)}
            e["prints"] = prints
            blk["methods"][m] = e
        x = blk["methods"][DEADAHEAD]
        for m in OTHERS:
            y = blk["methods"][m]
            if x["not_computed"] or y["not_computed"]:
                blk["contrasts"][f"{DEADAHEAD} - {m}"] = {"not_computed": x["not_computed"] or y["not_computed"]}
            else:
                blk["contrasts"][f"{DEADAHEAD} - {m}"] = paired_both(x["primary"]["nine_folds"]["per_group"],
                                                                     y["primary"]["nine_folds"]["per_group"])
        out[c] = blk
    return out


def sentence_values(routes: dict) -> dict:
    """The values the paper quotes (nine-group macros), each with its group count and the chance over the same
    groups: x = DeadAhead, y = chance, z = the image survival baseline, w = elapsed time; suffix L lytic, A
    apoptosis."""
    out = {}
    for c, k in ROUTE_SUFFIX.items():
        ch = routes[c]["chance"]["nine_folds"]
        out[f"y_{k}"] = {"what": "chance", "route": c, "value": ch["macro"], "n_groups": ch["n_groups"],
                         "groups_left_out": ch["groups_left_out"]}
        for slot, m in SLOTS.items():
            e = routes[c]["methods"][m]
            if e["not_computed"]:
                out[f"{slot}_{k}"] = {"what": m, "route": c, "value": None, "not_computed": e["not_computed"]}
                continue
            nf = e["primary"]["nine_folds"]
            out[f"{slot}_{k}"] = {"what": m, "route": c, "value": nf["macro"], "n_groups": nf["n_groups"],
                                  "groups_left_out": nf["groups_left_out"],
                                  "chance_over_the_same_groups": e["primary"]["chance_over_the_same_groups"]["macro"]}
    return out


def part_gap(bundle: dict) -> dict:
    """The gap analysis on the survival population of the evaluation cohort."""
    if "landmarks" not in bundle or "cumulative_incidence" not in bundle or "movies" not in bundle:
        raise ValueError("the bundle holds no landmark rows, cumulative incidences or movie table: the gap cannot be "
                         "computed")
    index_cumulative_incidence(bundle)
    pop, removed = horizon_population(bundle)
    ctx = context(bundle, pop)
    routes = gap_block(ctx)
    return {"post_hoc": True,
            "population": {"n_rows": len(pop), "rows_id": rows_id(pop), "cohort_removed": removed},
            "window_min": {"early": H_EARLY, "late": H_LATE}, "sets": SET_TEXT,
            "scores": {"primary": PRIMARY_LABEL, "print_f120": F120_LABEL, "print_given_survival": GIVEN_LABEL},
            "methods": METHOD_LABELS, "routes": routes, "paper": {"values": sentence_values(routes)}}


def _nine(block) -> float:
    """The nine-group macro of a block, NaN for a score that was not computed."""
    return block["nine_folds"]["macro"] if block and "nine_folds" in block else float("nan")


def format_gap(res: dict) -> str:
    """The gap analysis as text: AUPRC of deaths 60 to 120 min ahead per method and cause under the primary score and
    the two scores beside it, and DeadAhead's contrasts."""
    lines = [("Post hoc gap analysis: AUPRC of deaths 60 to 120 min ahead, forecasts whose cell died within 60 min "
              "left out (nine-group macro)"), "",
             f"{'method':<32}{'lytic':>8}{'apoptosis':>11}{'F120 lytic':>12}{'F120 apopt.':>13}"
             f"{'given lytic':>13}{'given apopt.':>14}"]
    ch = {c: res["routes"][c]["chance"]["nine_folds"]["macro"] for c in CAUSES}
    lines.append(f"{'chance (prevalence)':<32}{ch['LYTIC']:>8.3f}{ch['APOPTOSIS']:>11.3f}")
    for m in METHOD_ORDER:
        v, p = {}, {}
        for c in CAUSES:
            e = res["routes"][c]["methods"][m]
            v[c] = _nine(e.get("primary"))
            p[c] = e.get("prints") or {}
        cells = zip((p["LYTIC"].get("f120"), p["APOPTOSIS"].get("f120"), p["LYTIC"].get("given_survival_to_t_plus_60"),
                     p["APOPTOSIS"].get("given_survival_to_t_plus_60")), (12, 13, 13, 14))
        beside = "".join(f"{_nine(b):>{w}.3f}" if _finite(_nine(b)) else f"{'':>{w}}" for b, w in cells)
        lines.append(f"{m:<32}{v['LYTIC']:>8.3f}{v['APOPTOSIS']:>11.3f}{beside}")
    lines += ["", f"F120: {F120_LABEL}", f"given: {GIVEN_LABEL}"]
    lines += ["", "DeadAhead minus each method: macro difference, groups ahead of 9, outcome"]
    for m in OTHERS:
        for c in ("LYTIC", "APOPTOSIS"):
            p = res["routes"][c]["contrasts"][f"{DEADAHEAD} - {m}"]
            if "nine_folds" not in p:
                lines.append(f"  {m:<30}{c:<11}{p.get('not_computed')}")
                continue
            p = p["nine_folds"]
            lines.append(f"  {m:<30}{c:<11}{p.get('macro_difference', float('nan')):+8.3f}"
                         f"{p.get('x_ahead', 0):>4d}/{p['n']}  {outcome_text(p)}")
    return "\n".join(lines)
