"""Reproduce, post hoc, the warnings of the image baseline (DINOv2 image features + handcrafted features, refit head)
under the same training-only alarm thresholds as DeadAhead's warnings.

The thresholds come from the baseline's cross-fitted inner predictions on the training groups' reference rows (each
outer group and seed has eight inner heads, one per training group); the held-out scores are its Table 1
predictions. DeadAhead's and elapsed time's values come from the warnings part.
"""

from __future__ import annotations

import csv
import io
import json
import statistics

import numpy as np

from deadahead.evaluation.aggregate import outcome_text, paired_both
from deadahead.evaluation.alarms import (NOMINAL, PRIMARY_RATE, threshold_and_apply, warning_summary,
                                         warning_unit)
from deadahead.evaluation.rows import CAUSES, GROUPS, SEEDS, inner_groups
from deadahead.reproduce.table1 import prediction_scores
from deadahead.reproduce.warning_times import (DEADAHEAD, SUSTAINED_CSV_X, TIME, part_warnings, rate_label,
                                          warning_population)

IMAGE = "dinov2_handcrafted|refit"
LAST_HOUR = "60"                     # sustained coverage in the last hour before death
IMAGE_LABEL = ("DINOv2 image features + handcrafted features, refit head (the Table 1 entry); held-out scores = its "
               "Table 1 predictions; inner scores = 216 cross-fitted heads (each trained on seven of the held-out "
               "group's eight training groups)")
#: the summary keys of the paper's sentence, with readable names
SENTENCE_KEYS = {"r_L": "achieved_rate_lytic", "x_L": "sustained_last_hour_lytic", "k": "groups_ahead_lytic",
                 "n": "n_groups"}
#: the series of Fig. 2 with its spread: (column name, method, cause)
SPREAD_SERIES = (("survival_lytic_r02", DEADAHEAD, "LYTIC"), ("image_lytic_r02", IMAGE, "LYTIC"),
                 ("clock_lytic_r02", TIME, "LYTIC"), ("survival_apop_r02", DEADAHEAD, "APOPTOSIS"),
                 ("image_apop_r02", IMAGE, "APOPTOSIS"), ("clock_apop_r02", TIME, "APOPTOSIS"))
SPREAD_STATS = ("", "_min", "_max", "_q1", "_q3")


def roundtrip(obj):
    """The object as its JSON file holds it, so keys and floats compare like for like."""
    from deadahead.reproduce.main import jsonable
    return json.loads(json.dumps(obj, default=jsonable))


def index_image_predictions(bundle: dict) -> None:
    """Split the image baseline's held-out and inner prediction tables once by unit."""
    if "img_outer" not in bundle:
        p = bundle["predictions"][IMAGE]
        bundle["img_outer"] = {g: d for g, d in p.groupby("group")}
    if "img_inner" not in bundle:
        q = bundle["inner_predictions_by_method"][IMAGE]
        bundle["img_inner"] = {(g, int(s), h): d for (g, s, h), d in
                               q.groupby(["outer_group", "seed", "inner_group"])}


def image_ref_scores(bundle: dict, g: str, s: int, rows, ref_mask, cause: str):
    """The pooled inner scores of outer (g, s) on the eight training groups' reference rows, in fold order."""
    parts = []
    m = np.asarray(ref_mask, bool)
    for h in inner_groups(g):
        rh = rows[(rows["group"] == h).to_numpy() & m]
        parts.append(prediction_scores(bundle["img_inner"][(g, s, h)], s, rh)[cause])
    return np.concatenate(parts) if parts else np.zeros(0)


def image_models(bundle: dict, rows) -> dict:
    """Per held-out group g, seed s, cause c and nominal rate: the threshold from the pooled inner scores on the
    reference rows, applied once to g's held-out scores at seed s; thresholds and warning summaries."""
    neg = rows["negative"].to_numpy(bool)
    out = {"thresholds": {}, "models": {}}
    units = {}
    for g in GROUPS:
        gm = (rows["group"] == g).to_numpy()
        rg = rows[gm].reset_index(drop=True)
        for c in CAUSES:
            for rate, a in NOMINAL.items():
                for s in SEEDS:
                    try:
                        ref = image_ref_scores(bundle, g, s, rows, neg, c)
                        te = prediction_scores(bundle["img_outer"][g], s, rg)[c]
                    except ValueError as e:
                        raise ValueError(f"{IMAGE} {g} s{s} {c}: {e} (never partially scored)") from None
                    flags, thr = threshold_and_apply(ref, te, a)
                    out["thresholds"][f"{IMAGE}|{g}|s{s}|{c}|{rate}"] = thr
                    units.setdefault((c, rate), {})[(g, s)] = warning_unit(flags, rg, c, a)
    for (c, rate), u in units.items():
        out["models"].setdefault(IMAGE, {}).setdefault(c, {})[rate] = {
            **warning_summary(u), "units": {f"{g}|s{s}": v for (g, s), v in u.items()}}
    return roundtrip(out)


def image_contrasts(img: dict, base: dict) -> dict:
    """At every lead time X: DeadAhead minus the image baseline and the image baseline minus elapsed time (nine
    groups and the eight-group print), with the eligible deaths checked equal to DeadAhead's and elapsed time's."""
    out, checked = {}, 0
    for c in CAUSES:
        for rate in NOMINAL:
            b = base["models"][DEADAHEAD][c][rate]["sustained"]
            t = base["models"][TIME][c][rate]["sustained"]
            i = img["models"][IMAGE][c][rate]["sustained"]
            if set(i) != set(b) or set(b) != set(t):
                raise ValueError(f"{c} {rate}: the X sets differ ({sorted(i)}, {sorted(b)}, {sorted(t)})")
            for X in sorted(b, key=int):
                e_i, e_b, e_t = (v[X]["eligible_deaths_per_group"] for v in (i, b, t))
                if not e_i == e_b == e_t:
                    raise ValueError(f"{c} {rate} X={X}: the eligible deaths differ from DeadAhead's or elapsed time's")
                checked += 1
                pi = i[X]["nine_folds"]["per_group"]
                out[f"{DEADAHEAD} - {IMAGE}|{c}|{rate}|sustained {X}"] = paired_both(b[X]["nine_folds"]["per_group"],
                                                                                    pi)
                out[f"{IMAGE} - {TIME}|{c}|{rate}|sustained {X}"] = paired_both(pi, t[X]["nine_folds"]["per_group"])
    return {"contrasts": roundtrip(out), "eligible_deaths_check": {
        "passes": True, "n_route_rate_X_checked": checked}}


def sentence_inputs(img: dict, contrasts: dict) -> dict:
    """The values the paper quotes at a target of 0.2 alarms per cell-hour: the achieved pooled lytic rate, the lytic
    sustained coverage in the last hour (X = 60, mean over groups), the groups where DeadAhead is ahead of the image
    baseline there, and the achieved pooled rates of both causes."""
    rate = PRIMARY_RATE
    m = {c: img["models"][IMAGE][c][rate] for c in CAUSES}
    k = contrasts[f"{DEADAHEAD} - {IMAGE}|LYTIC|{rate}|sustained {LAST_HOUR}"]["nine_folds"]
    return {"rate": rate,
            "r_L": m["LYTIC"]["achieved_rate_exposure_weighted"]["nine_folds"]["rate_per_cell_hour"],
            "x_L": m["LYTIC"]["sustained"][LAST_HOUR]["nine_folds"]["macro"],
            "k": k.get("x_ahead"), "n": k.get("n"),
            "caption_achieved_pooled_rate_per_cell_hour": {
                c: m[c]["achieved_rate_exposure_weighted"]["nine_folds"]["rate_per_cell_hour"] for c in CAUSES}}


def part_image_warnings(bundle: dict, base: dict | None = None) -> dict:
    """The image baseline's warnings on the warning rows, and its contrasts with DeadAhead and elapsed time;
    ``base`` is the warnings part's result (computed when not given)."""
    if "inner_predictions_by_method" not in bundle or IMAGE not in bundle["inner_predictions_by_method"]:
        raise ValueError("the bundle holds no inner predictions of the image baseline")
    index_image_predictions(bundle)
    base = roundtrip(base if base is not None else part_warnings(bundle))
    rows, info = warning_population(bundle)
    img = image_models(bundle, rows)
    con = image_contrasts(img, base)
    sentence = {SENTENCE_KEYS.get(k, k): v for k, v in sentence_inputs(img, con["contrasts"]).items()}
    return {"post_hoc": True, "population": base["population"],
            "rows": {"n": len(rows), "reference_rows": int(rows["negative"].sum()),
                     "cohort_removed": info["cohort_removed"]},
            "method": {"name": IMAGE, "label": IMAGE_LABEL},
            "thresholds": img["thresholds"], "models": img["models"], **con, "sentence_inputs": sentence}


def fig2_table(warn: dict, image: dict | None = None) -> list[dict]:
    """The warning curves of Fig. 2: sustained coverage (percent of deaths alarmed at least X min ahead, nine-group
    macro, one decimal) of DeadAhead and elapsed time at both rates (``warn``: the warnings part) and of the image
    baseline at a target of 0.2 alarms per cell-hour (``image``: the image-warnings part, when given)."""
    series = [(DEADAHEAD, warn["models"][DEADAHEAD], ("0.2", "0.4"))]
    if image is not None:
        series.append((IMAGE.split("|")[0], image["models"][IMAGE], ("0.2",)))
    series.append((TIME, warn["models"][TIME], ("0.2", "0.4")))
    out = []
    for X in SUSTAINED_CSV_X:
        row = {"X_min": X}
        for name, models, rates in series:
            for c in ("APOPTOSIS", "LYTIC"):
                for r in rates:
                    v = models[c][f"{r}_per_cell_hour"]["sustained"][X]["nine_folds"]["macro"]
                    row[f"{name}_{c.lower()}_rate{r}"] = f"{100 * v:.1f}"
        out.append(row)
    return out


def fig2_csv(warn: dict, image: dict | None = None) -> str:
    """``fig2_table`` as CSV text."""
    rows = fig2_table(warn, image)
    head = list(rows[0])
    return "\n".join([",".join(head)] + [",".join(r[k] for k in head) for r in rows]) + "\n"


def spread_table(warn: dict, image: dict) -> list[dict]:
    """Fig. 2 with its spread: per X and series at 0.2 alarms per cell-hour, 100 x the mean, smallest, largest, lower
    and upper quartile (inclusive method) of the nine per-group shares, each already the mean over the group's seeds,
    to one decimal. ``warn`` and ``image``: the warnings and image-warnings parts."""
    source = {DEADAHEAD: warn, IMAGE: image, TIME: warn}
    rows = []
    for x in SUSTAINED_CSV_X:
        row = {"X": x}
        for name, m, c in SPREAD_SERIES:
            pg = source[m]["models"][m][c][PRIMARY_RATE]["sustained"][x]["nine_folds"]["per_group"]
            if len(pg) != len(GROUPS):
                raise ValueError(f"{name} at X = {x}: {len(pg)} groups contribute, not {len(GROUPS)}")
            vals = [pg[g] for g in sorted(pg)]
            q1, _, q3 = statistics.quantiles(vals, n=4, method="inclusive")
            for suffix, v in zip(SPREAD_STATS, (statistics.fmean(vals), min(vals), max(vals), q1, q3)):
                row[name + suffix] = f"{100 * v:.1f}"
        rows.append(row)
    return rows


def spread_csv(warn: dict, image: dict) -> str:
    """``spread_table`` as CSV text (the input of ``deadahead plot-warnings``)."""
    out = io.StringIO(newline="")
    w = csv.DictWriter(out, fieldnames=["X"] + [n + s for n, _, _ in SPREAD_SERIES for s in SPREAD_STATS])
    w.writeheader()
    w.writerows(spread_table(warn, image))
    return out.getvalue()


def format_image_warnings(res: dict) -> str:
    """The image baseline's warnings as text, and DeadAhead minus the image baseline at every lead time."""
    lines = [("Post hoc warnings of the image baseline (DINOv2 image features + handcrafted features): deaths warned X "
              "min ahead (%, mean over the 9 groups), training-only thresholds"), ""]
    for rate in NOMINAL:
        for c in ("LYTIC", "APOPTOSIS"):
            b = res["models"][IMAGE][c][rate]
            ach = b["achieved_rate_exposure_weighted"]["nine_folds"]["rate_per_cell_hour"]
            cov = "  ".join(f"{X}:{100 * b['sustained'][X]['nine_folds']['macro']:5.1f}" for X in SUSTAINED_CSV_X)
            lines.append(f"  {rate_label(rate)}, {c.lower():<10} achieved {ach:.2f} pooled   {cov}")
    lines += ["", ("DeadAhead minus the image baseline, deaths warned X min ahead: macro difference (percentage "
                   "points), groups ahead of 9, outcome")]
    for rate in NOMINAL:
        for c in ("LYTIC", "APOPTOSIS"):
            for X in SUSTAINED_CSV_X:
                p = res["contrasts"][f"{DEADAHEAD} - {IMAGE}|{c}|{rate}|sustained {X}"]["nine_folds"]
                lines.append(f"  {rate_label(rate):<12}{c.lower():<11}X={X:<4}"
                             f"{100 * p.get('macro_difference', float('nan')):+7.1f}"
                             f"{p.get('x_ahead', 0):>4d}/{p['n']}  {outcome_text(p)}")
    return "\n".join(lines)
