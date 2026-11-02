"""Reproduce the fixed-horizon results: at h = 30, 60, 120 and 240 min, DeadAhead and elapsed time at training-only
alarm thresholds (precision, sensitivity, AUPRC, prevalence), the AUPRC of the other survival methods, and the
survival estimators of the four survival methods (IPCW Brier score, IPA, Wolbers' concordance, E/O, the IPCW
calibration intercept and slope).

Population: the landmark rows of the survival population on the evaluation cohort, with complete-movie follow-up.
At each horizon the outcome sets are P (a death of the cause by t + h), Q (a death of the other cause by t + h),
N (no death by t + h and followed through it) and U (left out). The reference rows of the thresholds are the
training groups' N rows (event-free through t + h and followed through it), scored by DeadAhead's cross-fitted
inner heads at F_c(h).
"""

from __future__ import annotations

import json

import numpy as np

from deadahead.evaluation.aggregate import both_folds, paired_both
from deadahead.evaluation.alarms import (CUT_FRAME, NOMINAL, PRIMARY_RATE, reference_record, threshold_and_apply,
                                         truncate_at_cut)
from deadahead.evaluation.horizons import (fixed_horizon_summary, fixed_horizon_unit, outcome_sets,
                                           precision_sensitivity)
from deadahead.evaluation.rows import CAUSES, GROUPS, KEY, SEEDS, align_scores, inner_groups, key_index
from deadahead.evaluation.survival_metrics import _jsonable, _scorable, _train, metrics_block
from deadahead.evaluation.table1 import auprc_sets
from deadahead.reproduce.gap import (DEADAHEAD, DEEPHIT, HANDCRAFTED_SURVIVAL, IMAGE_SURVIVAL, dh_not_run_groups,
                                     horizon_population, index_cumulative_incidence)
from deadahead.reproduce.warning_times import TIME, TIME_SEED, index_predictions, time_for

HORIZONS = (30, 60, 120, 240)
INNER_FILE_HORIZONS = (60, 120, 240)            # the inner table at 30 min is the warnings' inner_predictions
SET_NAME = "population"                         # the key of the population in metrics_block's JSON form
#: ipa_training_null: IPA against the Aalen-Johansen incidence of the training groups (pre-specified beside the
#: held-out group's own incidence, which "ipa" uses)
SURV_KEYS = ("ipa", "ipa_training_null", "brier_ipcw", "c_no_same_track", "e_over_o", "ipcw_slope",
             "ipcw_intercept")
#: the methods with cumulative-incidence outputs scored beside DeadAhead (AUPRC, prevalence and the survival
#: estimators; no alarm threshold, as only DeadAhead has cross-fitted inner heads)
CIF_MODELS = {HANDCRAFTED_SURVIVAL: "the handcrafted features under the survival head",
              IMAGE_SURVIVAL: "DINOv2 image features + handcrafted features under the same survival head",
              DEEPHIT: "DeepHit on the DeadAhead inputs"}
POPULATION_LABEL = ("the landmark rows of the survival population on the evaluation cohort; complete-movie "
                    "follow-up; reference rows at h: the training groups' rows event-free through t + h and followed "
                    "through it")


# ------------------------------------------------------------------ the stored predictions
def index_horizon_predictions(bundle: dict) -> None:
    """Join the stored cumulative incidences at 60 and 120 min with those at 30 and 240 min per (method, group,
    seed), and split DeadAhead's inner predictions at 60, 120 and 240 min by (outer group, seed, inner group)."""
    index_cumulative_incidence(bundle)
    if "hz_by_unit" not in bundle:
        extra = {(m, g, int(s)): d for (m, g, s), d in
                 bundle["cumulative_incidence_30_240"].groupby(["method", "group", "seed"])}
        if set(extra) != set(bundle["ci_by_unit"]):
            raise ValueError("the two cumulative-incidence tables hold different (method, group, seed) units")
        hz = {}
        for u, a in bundle["ci_by_unit"].items():
            b = extra[u]
            if not key_index(a).equals(key_index(b)):
                raise ValueError(f"{u}: the two cumulative-incidence tables hold different rows")
            cols = [c for c in b.columns if c.startswith("F_")]
            hz[u] = a.assign(**{c: b[c].to_numpy() for c in cols})
        bundle["hz_by_unit"] = hz
    if "inner_hz_by_unit" not in bundle:
        q = bundle["inner_cumulative_incidence"]
        bundle["inner_hz_by_unit"] = {(g, int(s), h): d for (g, s, h), d in
                                      q.groupby(["outer_group", "seed", "inner_group"])}


def cif_preds(bundle: dict, model: str, g: str, s: int, pg, horizons) -> dict:
    """{h: {cause: array over pg}} of a method's stored F_k(t, h) at (g, s), h in 30, 60, 120, 240; nan where the
    method has no row."""
    z = bundle["hz_by_unit"].get((model, g, int(s)))
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


def held_out_scores(bundle: dict, g: str, s: int, cause: str, h: int, rows):
    """DeadAhead's held-out F_c(h) of outer (g, s) at ``rows``' keys (raises where a row has no finite score)."""
    z = bundle["hz_by_unit"][(DEADAHEAD, g, int(s))]
    return align_scores(z, z[f"F_{cause}_{h}"].to_numpy(float), rows)


def inner_ref_scores(bundle: dict, g: str, s: int, rows, ref_mask, cause: str, h: int):
    """The pooled cross-fitted inner F_c(h) of outer (g, s) on the training groups' reference rows."""
    parts = []
    m = np.asarray(ref_mask, bool)
    for hh in inner_groups(g):
        rh = rows[(rows["group"] == hh).to_numpy() & m]
        if h == 30:
            z, col = bundle["inner_by_unit"][(g, s, hh, cause)], "score"
        else:
            z, col = bundle["inner_hz_by_unit"][(g, s, hh)], f"F_{cause}_{h}"
        parts.append(align_scores(z, z[col].to_numpy(float), rh))
    return np.concatenate(parts) if parts else np.zeros(0)


# ------------------------------------------------------------------ the survival estimators
def survival_metrics(bundle: dict, pop, g: str, s: int, horizons, model: str = DEADAHEAD) -> dict:
    """The survival estimators of one method at (g, s), each None where it is not scorable."""
    pg = pop[pop["group"] == g].reset_index(drop=True)
    preds = cif_preds(bundle, model, g, s, pg, horizons)
    blk = metrics_block(pg, preds, horizons, _train(pop, g), ibs_too=False)
    fr = json.loads(json.dumps({"primary": {SET_NAME: blk}}, default=_jsonable))
    out = {}
    for c in ("APOPTOSIS", "LYTIC"):
        for h in horizons:
            out[(c, int(h))] = {k: _scorable(fr, SET_NAME, c, int(h), k) for k in SURV_KEYS}
    return out


def surv_summary(surv: dict, c: str, h: int) -> dict:
    """{(g, s): survival_metrics} -> the nine- and eight-group macros of each estimator, and the groups with IPA
    below zero."""
    blk = {}
    for k in SURV_KEYS:
        pgs = {g: {s: (float("nan") if v[(c, h)][k] is None else v[(c, h)][k]) for (gg, s), v in surv.items()
                   if gg == g} for g in sorted({g for g, _ in surv})}
        blk[k] = both_folds(pgs)
    ipa = blk["ipa"]["nine_folds"]["per_group"]
    blk["ipa_groups_below_zero"] = {"n_below_zero": int(sum(v < 0 for v in ipa.values())), "of": len(ipa),
                                    "groups": sorted(g for g, v in ipa.items() if v < 0)}
    return blk


def cif_auprc(bundle: dict, model: str, pop, h: int) -> dict:
    """AUPRC and prevalence of a cumulative-incidence method at F_c(h) on the determined rows (P, Q, N), per seed
    and group; a determined row without a finite prediction raises ValueError."""
    out = {}
    skip = dh_not_run_groups(bundle) if model == DEEPHIT else set()
    for c in CAUSES:
        pa, pv = {}, {}
        for g in [x for x in GROUPS if x not in skip]:
            pg = pop[pop["group"] == g].reset_index(drop=True)
            sets = outcome_sets(pg["T_min"], pg["delta"], pg["C_min"], c, h)
            det = sets["P"] | sets["Q"] | sets["N"]
            for s in SEEDS:
                p = cif_preds(bundle, model, g, s, pg, (h,))[h][c]
                if not np.all(np.isfinite(p[det])):
                    raise ValueError(f"{model} {g} s{s}: a determined row has no finite prediction at h{h}")
                pa.setdefault(g, {})[s] = auprc_sets(p, sets)
                pv.setdefault(g, {})[s] = precision_sensitivity(np.zeros(len(pg), bool), sets)["prevalence"]
        out[c] = {"auprc": both_folds(pa), "prevalence": both_folds(pv)}
    return out


# ------------------------------------------------------------------ the fixed-horizon table
def horizons_block(bundle: dict, pops: dict, thresholds_from: dict | None = None) -> dict:
    """pops: {h: population}. Per cause and horizon: DeadAhead and elapsed time at their F_c(h) and time-score
    thresholds on the training groups' N rows (reselected unless ``thresholds_from`` gives the reference populations
    to reuse), the AUPRC of the other survival methods, the survival estimators and the paired contrasts."""
    out = {"models": {}, "reference_populations": {}, "survival_estimators": {}, "contrasts": {}}
    for h, pop in pops.items():
        ref_pop = pop if thresholds_from is None else thresholds_from[h]
        N_ref = outcome_sets(ref_pop["T_min"], ref_pop["delta"], ref_pop["C_min"], "LYTIC", h)["N"]
        ts_ref = time_for(ref_pop)
        ts_pop = time_for(pop, None if thresholds_from is None else ref_pop)
        out["reference_populations"][str(h)] = {g: reference_record(ref_pop, N_ref, g) for g in GROUPS}
        surv = {}
        for c in CAUSES:
            units = {}
            for g in GROUPS:
                gm = (pop["group"] == g).to_numpy()
                pg = pop[gm].reset_index(drop=True)
                sets = outcome_sets(pg["T_min"], pg["delta"], pg["C_min"], c, h)
                trm = (ref_pop["group"] != g).to_numpy()
                for rate, a in NOMINAL.items():
                    for s in SEEDS:
                        ref = inner_ref_scores(bundle, g, s, ref_pop, N_ref, c, h)
                        # only the determined rows (P, Q, N) need a score: U is excluded from every quantity here,
                        # and the population holds landmarks no head predicts that are never determined; they stay
                        # unscored and unalarmed
                        det = sets["P"] | sets["Q"] | sets["N"]
                        sc = np.full(len(pg), np.nan)
                        sc[det] = held_out_scores(bundle, g, s, c, h, pg[det])
                        flags = np.zeros(len(pg), bool)
                        flags[det], thr = threshold_and_apply(ref, sc[det], a)
                        units.setdefault((DEADAHEAD, rate), {})[(g, s)] = {**fixed_horizon_unit(flags, sc, sets),
                                                                           "threshold": thr}
                    flags, thr = threshold_and_apply(ts_ref[trm & N_ref], ts_pop[gm], a)
                    units.setdefault((TIME, rate), {})[(g, TIME_SEED)] = {
                        **fixed_horizon_unit(flags, ts_pop[gm], sets), "threshold": thr}
            for (m, rate), u in units.items():
                out["models"].setdefault(m, {}).setdefault(c, {}).setdefault(str(h), {})[rate] = {
                    **fixed_horizon_summary(u), "units": {f"{g}|s{s}": v for (g, s), v in u.items()}}
            for rate in NOMINAL:
                for k in ("precision", "sensitivity"):
                    b = out["models"][DEADAHEAD][c][str(h)][rate][k]["nine_folds"]["per_group"]
                    t_ = out["models"][TIME][c][str(h)][rate][k]["nine_folds"]["per_group"]
                    out["contrasts"][f"{DEADAHEAD} - {TIME}|{c}|h{h}|{rate}|{k}"] = paired_both(b, t_)
        for m in (DEADAHEAD, *CIF_MODELS):
            skip = dh_not_run_groups(bundle) if m == DEEPHIT else set()
            try:
                surv = {(g, s): survival_metrics(bundle, pop, g, s, (h,), m) for g in GROUPS for s in SEEDS
                        if g not in skip}
                if m != DEADAHEAD:
                    for c, v in cif_auprc(bundle, m, pop, h).items():
                        out.setdefault("cif_models_auprc", {}).setdefault(m, {}).setdefault(c, {})[str(h)] = v
            except ValueError as e:
                if m == DEADAHEAD:
                    raise
                out.setdefault("not_computed", {})[f"{m}|h{h}"] = str(e)
                continue
            for c in CAUSES:
                out["survival_estimators"].setdefault(m, {}).setdefault(c, {})[str(h)] = surv_summary(surv, c, h)
        for c in CAUSES:
            for m in CIF_MODELS:
                ca = (out.get("cif_models_auprc") or {}).get(m, {}).get(c, {}).get(str(h))
                if ca:
                    out["contrasts"][f"{DEADAHEAD} - {m}|{c}|h{h}|auprc"] = paired_both(
                        out["models"][DEADAHEAD][c][str(h)][PRIMARY_RATE]["auprc"]["nine_folds"]["per_group"],
                        ca["auprc"]["nine_folds"]["per_group"])
                if m in out["survival_estimators"] and str(h) in out["survival_estimators"][m].get(c, {}):
                    for k in ("ipa", "c_no_same_track"):
                        out["contrasts"][f"{DEADAHEAD} - {m}|{c}|h{h}|{k}"] = paired_both(
                            out["survival_estimators"][DEADAHEAD][c][str(h)][k]["nine_folds"]["per_group"],
                            out["survival_estimators"][m][c][str(h)][k]["nine_folds"]["per_group"])
    out["cif_models"] = CIF_MODELS
    return out


def has_inputs(bundle: dict) -> bool:
    """The bundle holds the input tables of the fixed-horizon results."""
    return all(k in bundle for k in ("inner_predictions", "movies", "landmarks", "cumulative_incidence",
                                     "cumulative_incidence_30_240", "inner_cumulative_incidence"))


def prepare(bundle: dict) -> None:
    """Check the inputs and index the stored predictions once."""
    if not has_inputs(bundle):
        raise ValueError("the bundle holds no stored cumulative incidences at 30 to 240 min or no inner predictions "
                         "at 60 to 240 min: the fixed-horizon results cannot be computed")
    index_predictions(bundle)
    index_horizon_predictions(bundle)


def part_horizons(bundle: dict) -> dict:
    """The fixed-horizon results on the survival population at h = 30, 60, 120 and 240 min."""
    prepare(bundle)
    pop, removed = horizon_population(bundle)
    return {"population": POPULATION_LABEL, "cohort_removed": removed, "n_rows": len(pop),
            **horizons_block(bundle, {h: pop for h in HORIZONS})}


def common_cut_population(bundle: dict, h: int):
    """The population restricted to rows with t + h/3 <= 438, follow-up and outcomes truncated at frame 438."""
    pop, _ = horizon_population(bundle)
    pop = truncate_at_cut(pop[pop["t_obs_frame"] + h / 3.0 <= CUT_FRAME].reset_index(drop=True))
    return pop.sort_values(KEY).reset_index(drop=True)


def part_common_cut(bundle: dict) -> dict:
    """The fixed-horizon results at the common cut: rows t + h/3 <= 438, outcomes truncated at 438, the survival
    estimators' weights and reference risks re-estimated on that population, the alarm thresholds of the main
    analysis (its reference populations)."""
    prepare(bundle)
    full, _ = horizon_population(bundle)
    pops = {h: common_cut_population(bundle, h) for h in HORIZONS}
    return {"rows": "t + h/3 <= 438, labels truncated at 438; the survival estimators' weights and reference risks "
                    "re-estimated on that population; thresholds: the main analysis' reference populations",
            "n_rows": {str(h): len(p) for h, p in pops.items()},
            **horizons_block(bundle, pops, thresholds_from={h: full for h in HORIZONS})}


def format_horizons(res: dict) -> str:
    """The fixed-horizon results as text: per cause and horizon, prevalence, the achieved rate, precision and
    sensitivity at a target of 0.2 alarms per cell-hour, AUPRC, IPA against both reference incidences with the groups
    below zero, concordance, the calibration slope and predicted over observed incidence of each method (nine-group
    macros)."""
    title = ("Fixed horizons: nine-group macros; prevalence (%); rate (achieved, mean over groups), precision and "
             "sensitivity at a target of 0.2 alarms per event-free reference cell-hour (DeadAhead and elapsed time; "
             "each horizon its own threshold); IPA against the held-out group's incidence and, IPA (train), against "
             "the training groups' incidence; <0: groups with IPA below zero; slope: the calibration slope "
             "(ipcw_slope); P/O: predicted over observed incidence (e_over_o)")
    header = (f"{'cause':<10}{'h':>4}  {'method':<28}{'Prev.':>7}{'rate':>6}{'precision':>10}{'sensitivity':>12}"
              f"{'AUPRC':>8}{'IPA':>8}{'IPA (train)':>12}{'<0':>4}{'C':>7}{'slope':>7}{'P/O':>7}")
    lines = [title, "", header]
    est = res["survival_estimators"]
    cif = res.get("cif_models_auprc", {})
    for c in ("LYTIC", "APOPTOSIS"):
        for h in map(str, HORIZONS):
            for m in (DEADAHEAD, TIME, *CIF_MODELS):
                prev = rate = prec = sens = auprc = ipa = ipa_tr = below = conc = slope = eo = float("nan")
                if m in res["models"]:
                    b = res["models"][m][c][h][PRIMARY_RATE]
                    prev = 100 * b["prevalence"]["nine_folds"]["macro"]
                    rate = b["achieved_rate_equal_group_macro"]["nine_folds"]["macro"]
                    prec, sens = b["precision"]["nine_folds"]["macro"], b["sensitivity"]["nine_folds"]["macro"]
                    auprc = b["auprc"]["nine_folds"]["macro"]
                elif m in cif and h in cif[m].get(c, {}):
                    prev = 100 * cif[m][c][h]["prevalence"]["nine_folds"]["macro"]
                    auprc = cif[m][c][h]["auprc"]["nine_folds"]["macro"]
                if m in est and h in est[m].get(c, {}):
                    e = est[m][c][h]
                    ipa, ipa_tr = e["ipa"]["nine_folds"]["macro"], e["ipa_training_null"]["nine_folds"]["macro"]
                    below = e["ipa_groups_below_zero"]["n_below_zero"]
                    conc = e["c_no_same_track"]["nine_folds"]["macro"]
                    slope, eo = e["ipcw_slope"]["nine_folds"]["macro"], e["e_over_o"]["nine_folds"]["macro"]
                vals = ((prev, 7, 1), (rate, 6, 2), (prec, 10, 3), (sens, 12, 3), (auprc, 8, 3), (ipa, 8, 3),
                        (ipa_tr, 12, 3), (below, 4, 0), (conc, 7, 3), (slope, 7, 3), (eo, 7, 3))
                cells = "".join(f"{v:>{w}.{d}f}" if np.isfinite(v) else f"{'':>{w}}" for v, w, d in vals)
                lines.append(f"{c.lower():<10}{h:>4}  {m:<28}{cells}")
    lines += ["", "DeadAhead minus each method: macro difference, groups ahead of 9"]
    for m in CIF_MODELS:
        for c in ("LYTIC", "APOPTOSIS"):
            for h in HORIZONS:
                cells = []
                for key, name in (("auprc", "AUPRC"), ("ipa", "IPA"), ("c_no_same_track", "C")):
                    v = res.get("contrasts", {}).get(f"{DEADAHEAD} - {m}|{c}|h{h}|{key}")
                    if v:
                        p = v["nine_folds"]
                        cells.append(f"{name} {p.get('macro_difference', float('nan')):+.3f} "
                                     f"{p.get('x_ahead', 0)}/{p['n']}")
                lines.append(f"{c.lower():<10}{h:>4}  {m:<28}" + "   ".join(cells))
    return "\n".join(lines)
