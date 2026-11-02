"""Scores of the yeast death probe from its stored predictions: VideoMAE fine-tuned on six positions (from two
starting points) against the best simple rule on each of the 7 held-out positions, at 30 min.

Reading on the 7 position differences Delta_p: AHEAD needs mean Delta >= +0.02 and Delta_p > 0 on at least 6 of 7;
NO TANGIBLE DIFFERENCE when |mean| < 0.02; BEHIND needs mean <= -0.02 and Delta_p < 0 on at least 6; otherwise NOT
CONSISTENT. The unit is the position, n = 7.

The bundle holds labels/ (the human classes, the picture tables and the per-fold inputs), predictions/ (the held-out
predictions of every run) and, optionally, survival/ (the survival layer of every run: its AUPRC at every horizon,
lead time, threshold-free survival scores and next-bud split). The rules are refitted from labels/. In names and result
keys, ``bar`` is the best simple rule.

    deadahead yeast-score [--bundle <yeast bundle>] [--out scores.json] [--no-sensitivity] [--no-lead-time]
                          [--no-verify]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from deadahead.paths import data_root

from . import constants as C
from . import data as D
from . import lifespans as L

BAND = 0.02
NEED = 6                                    # 6 of 7 positions
SIGN_FLIP_FLOOR = 2 / 2 ** 7                # 0.0156
CLASS_NAMES = {
    "bar": ("BEATS THE BEST RULE", "NO TANGIBLE DIFFERENCE", "THE BEST RULE WINS", "NOT CONSISTENT"),
    "transfer": ("FINE-TUNED START AHEAD", "NO TANGIBLE DIFFERENCE", "PUBLIC START AHEAD", "NOT CONSISTENT"),
}
COMPARISON_NAMES = {"comparison_A": "fine-tuned start against the best rule",
                    "comparison_A2_secondary": "public start against the best rule",
                    "comparison_B": "fine-tuned start against the public start"}


def default_bundle() -> Path:
    return data_root() / "yeast_bundle"


# ------------------------------------------------------------------ the reading
def region(m: float) -> int:
    return 1 if m >= BAND else (-1 if m <= -BAND else 0)


def sign_flip_p(deltas) -> float:
    """Exact two-sided sign-flip p over the position differences (floor 2 / 2^n)."""
    d = [float(x) for x in deltas]
    obs = abs(sum(d))
    n_ext = sum(1 for s in itertools.product((1, -1), repeat=len(d))
                if abs(sum(si * di for si, di in zip(s, d))) >= obs - 1e-15)
    return n_ext / 2 ** len(d)


def classify(deltas: dict, kind: str = "bar") -> dict:
    """The reading on {position: Delta_p}, with the leave-one-position-out means and the positions whose removal
    moves the mean across a band edge."""
    names = CLASS_NAMES[kind]
    vals = [float(deltas[p]) for p in C.POSITIONS if p in deltas]
    if len(vals) != len(C.POSITIONS):
        raise ValueError(f"a reading needs all 7 positions, got {len(vals)}")
    mean = sum(vals) / len(vals)
    kp, km = sum(v > 0 for v in vals), sum(v < 0 for v in vals)
    if mean >= BAND and kp >= NEED:
        cls = names[0]
    elif abs(mean) < BAND:
        cls = names[1]
    elif mean <= -BAND and km >= NEED:
        cls = names[2]
    else:
        cls = names[3]
    jack = {p: (sum(vals) - float(deltas[p])) / (len(vals) - 1) for p in C.POSITIONS}
    fragile = [p for p in C.POSITIONS if region(jack[p]) != region(mean)]
    return {"class": cls, "mean_delta": mean, "k_plus": kp, "k_minus": km, "n": len(vals),
            "sign_flip_p": sign_flip_p(vals), "sign_flip_floor": SIGN_FLIP_FLOOR,
            "jackknife": {"means": jack, "min": min(jack.values()), "max": max(jack.values())},
            "fragile_positions": fragile, "deltas": {p: float(deltas[p]) for p in C.POSITIONS}}


def survival_layer_scorable(start: str, ok) -> dict:
    """All 7 positions with at least 2 of 3 runs, at most one position with only 2."""
    n = {p: sum(bool(ok(run_name(start, p, s))) for s in C.SEEDS) for p in C.POSITIONS}
    scorable = all(v >= 2 for v in n.values()) and sum(v == 2 for v in n.values()) <= 1
    return {"scorable": scorable, "runs_per_position": n}


# ------------------------------------------------------------------ per-position quantities
def per_position_auprc(labels, preds: dict) -> dict:
    """{model: {position: AUPRC}} on each position's landmarks with a determined 30-min label. ``labels``: unit,
    position, t, y_30; ``preds``: {model: frame with unit, t, p_death}. Every model scores the same keys."""
    from .metrics import auprc
    det = labels[labels[f"y_{C.PRIMARY_H}"].notna()]
    out = {}
    for name, pr in preds.items():
        pos = tuple(pr.attrs.get("positions", C.POSITIONS))           # a run scores its held-out position
        d = det[det["position"].isin(pos)]
        m = d.merge(pr[["unit", "t", "p_death"]], on=["unit", "t"], how="left", validate="one_to_one")
        if m["p_death"].isna().any() or len(m) == 0:
            raise ValueError(f"{name} has no score for {int(m['p_death'].isna().sum())} landmark(s) of {pos}")
        out[name] = {p: float(auprc(g[f"y_{C.PRIMARY_H}"].astype(int).to_numpy(), g["p_death"].to_numpy()))
                     for p, g in m.groupby("position")}
    return out


def row_means(per_model: dict, members: dict) -> dict:
    """a_R(p) = the mean over R's models (that scored p) of their AUPRC on p."""
    out = {}
    for p in C.POSITIONS:
        vals = [per_model[m][p] for m in members if p in per_model.get(m, {})]
        if vals:
            out[p] = sum(vals) / len(vals)
    return out


def aligned_scores(keys, frames: list) -> list:
    """Each prediction frame's p_death on ``keys`` (unit, t), in key order; every key must have a score."""
    out = []
    for fr in frames:
        m = keys[["unit", "t"]].merge(fr[["unit", "t", "p_death"]], on=["unit", "t"], how="left",
                                      validate="one_to_one")
        if m["p_death"].isna().any():
            raise ValueError("a model has no score for a key of the rules' rows")
        out.append(m["p_death"].to_numpy(float))
    return out


def position_intervals(bar_h: dict, labels, rows_frames: dict, h: int = C.PRIMARY_H) -> dict:
    """On one held-out position: 2,000 resamples of whole lifespans (seed 2601562), the best rule picked on the full
    position kept fixed; intervals of each row's a_R(p), of the best rule and of Delta_p."""
    from . import rules as B
    keys = bar_h["keys"]
    y = keys.merge(labels[["unit", "t", f"y_{h}"]], on=["unit", "t"], how="left")[f"y_{h}"].astype(int).to_numpy()
    scores = {r: aligned_scores(keys, frs) for r, frs in rows_frames.items() if frs}
    return B.bootstrap(y, scores, bar_h["scores"][bar_h["bar"]["member_key"]], keys["unit"].to_numpy())


def per_position_auroc(labels, preds: dict) -> dict:
    """AUROC per model and position."""
    from .metrics import auroc
    det = labels[labels[f"y_{C.PRIMARY_H}"].notna()]
    out = {}
    for name, pr in preds.items():
        pos = tuple(pr.attrs.get("positions", C.POSITIONS))
        m = det[det["position"].isin(pos)].merge(pr[["unit", "t", "p_death"]], on=["unit", "t"], how="left")
        out[name] = {p: float(auroc(g[f"y_{C.PRIMARY_H}"].astype(int).to_numpy(), g["p_death"].to_numpy()))
                     for p, g in m.groupby("position")}
    return out


def spread(per_model: dict, members: list, p: str) -> dict:
    v = np.array([per_model[m][p] for m in members if p in per_model.get(m, {})], dtype=float)
    return {"n": int(len(v)), "min": float(v.min()), "max": float(v.max()), "sd": float(v.std(ddof=1)) if len(v) > 1
            else None} if len(v) else {"n": 0}


def phase_groups(rows_k) -> dict:
    """From labels at or before t: before / after the first long cycle, or not yet decidable (under 7 buds so far);
    and the time-since-last-bud bins (under 1 h, 1-4 h, over 4 h)."""
    flag = rows_k["first_long_cycle"].to_numpy() == 1
    decid = rows_k["divisions"].to_numpy() >= 7
    phase = np.where(flag, "after_first_long_cycle", np.where(decid, "before_first_long_cycle", "not_yet_decidable"))
    sb = rows_k["since_last_bud_min"].to_numpy(float)
    bud = np.where(sb < 60, "under_1h", np.where(sb <= 240, "1h_to_4h", "over_4h"))
    return {"phase": phase, "since_last_bud": bud}


def split_by_phase(rows, bar: dict, row_frames: dict) -> dict:
    """AUPRC at 30 min in each phase and time-since-bud bin, per position and pooled over positions, for each row
    (the mean over its models) and for the best rule. ``row_frames``: {row: {position: [frames, one per model]}};
    a frame's ``attrs["model"]`` names its column (the seed), so a missing seed leaves that position empty."""
    from .metrics import auprc
    parts = []
    for p in C.POSITIONS:
        b = bar[p][C.PRIMARY_H]
        k = b["keys"]
        rk = k.merge(rows[["unit", "t", "y_30", "first_long_cycle", "divisions", "since_last_bud_min"]],
                     on=["unit", "t"], how="left")
        g = phase_groups(rk)
        fr = pd.DataFrame({"position": p, "y": rk["y_30"].astype(int).to_numpy(), "phase": g["phase"],
                           "since_last_bud": g["since_last_bud"], "bar": b["scores"][b["bar"]["member_key"]]})
        for r, by_pos in row_frames.items():
            frs = by_pos.get(p, [])
            for i, (f_, s_) in enumerate(zip(frs, aligned_scores(k, frs))):
                fr[f"{r}#{f_.attrs.get('model', i)}"] = s_
        parts.append(fr)
    d = pd.concat(parts, ignore_index=True)
    cols = ["bar"] + [c for c in d.columns if "#" in c]

    def ap(sub, c):
        sub = sub[sub[c].notna()]
        return float(auprc(sub["y"].to_numpy(), sub[c].to_numpy())) if sub["y"].sum() else None

    out = {}
    for split in ("phase", "since_last_bud"):
        res = {}
        for grp, sub in d.groupby(split):
            ent = {"n_landmarks": int(len(sub)), "n_positive": int(sub["y"].sum()), "pooled": {}, "per_position": {}}
            for r in ["bar", *row_frames]:
                cs = [c for c in cols if c == r or c.startswith(f"{r}#")]
                vals = [v for v in (ap(sub, c) for c in cs) if v is not None]
                ent["pooled"][r] = float(np.mean(vals)) if vals else None
                ent["per_position"][r] = {}
                for p, sp in sub.groupby("position"):
                    pv = [v for v in (ap(sp, c) for c in cs) if v is not None]
                    ent["per_position"][r][p] = float(np.mean(pv)) if pv else None
            res[grp] = ent
        out[split] = res
    return out


def sensitivity_series(series: dict, which: str) -> dict:
    """'empty_as_death' (lifespans ending empty become events at their first empty frame) or
    'clog_empty_left_out' (lifespans ending clogged or empty removed)."""
    out = {}
    for trap, cls in series.items():
        ls = L.detecdiv_lifespan(cls)
        end_cls = None if ls is None or ls[1] >= len(cls) else int(cls[ls[1]])
        if which == "empty_as_death":
            c = cls.copy()
            if end_cls == L.EMPTY:
                c[ls[1]] = L.DEAD
            out[trap] = c
        elif which == "clog_empty_left_out":
            if end_cls not in (L.EMPTY, L.CLOG):
                out[trap] = cls
        else:
            raise ValueError(which)
    return out


def sensitivity_rows(series: dict, pics: dict, rows_main, row_frames: dict, main_classes: dict) -> dict:
    """The same models re-scored on the changed labels (no retraining), the rules refitted on them (which favours
    the rules); a class that differs from the main reading is marked fragile."""
    from . import rules as B
    out = {}
    for which in ("empty_as_death", "clog_empty_left_out"):
        rows_s, notes = D.build_rows(sensitivity_series(series, which))
        idx = rows_main.reset_index()[["index", "unit", "t"]].merge(rows_s[["unit", "t"]].reset_index(),
                                                                      on=["unit", "t"], suffixes=("_m", "_s"))
        ent = {"n_deaths": notes["n_deaths"], "n_lifespans": notes["n_lifespans"], "readings": {}}
        bars = {}
        for p in C.POSITIONS:
            pic = pics[p].loc[idx["index_m"].to_numpy()].set_axis(rows_s.index[idx["index_s"].to_numpy()])
            bars[p] = B.bar_for_position(rows_s.loc[pic.index], pic, p, horizons=(C.PRIMARY_H,))[C.PRIMARY_H]
        labels_s = rows_s[["unit", "position", "t", *[f"y_{h}" for h in C.HORIZONS]]]
        for r, by_pos in row_frames.items():
            a = {}
            for p in C.POSITIONS:
                frs = by_pos.get(p, [])
                if not frs:
                    continue
                vals = []
                for fr in frs:
                    f2 = fr.copy()
                    f2.attrs["positions"] = (p,)
                    vals.append(per_position_auprc(labels_s, {"m": f2})["m"][p])
                a[p] = sum(vals) / len(vals)
            if len(a) == len(C.POSITIONS):
                rd = classify({p: a[p] - bars[p]["bar"]["auprc"] for p in C.POSITIONS}, "bar")
                ent["readings"][r] = {"class": rd["class"], "mean_delta": rd["mean_delta"], "k_plus": rd["k_plus"],
                                      "k_minus": rd["k_minus"],
                                      "fragile": main_classes.get(r) is not None and rd["class"] != main_classes[r]}
        out[which] = ent
    return out


# ------------------------------------------------------------------ lead time
def lifespan_facts(series: dict) -> tuple[dict, dict]:
    """{unit: event frame} for the deaths and {unit: bud frames} inside each lifespan."""
    events, buds = {}, {}
    for trap, cls in series.items():
        ls = L.detecdiv_lifespan(cls)
        if ls is None:
            continue
        start, end, event = ls
        if event:
            events[trap] = end
        buds[trap] = L.detecdiv_buds(cls, start, end)
    return events, buds


def lead_table_on(labels, keys, scores, thresholds: dict, events: dict, buds: dict) -> dict:
    """The lead-time table of one score on the rows ``keys`` (one position's rows with a determined 240-min
    status)."""
    from . import leadtime as LT
    g = keys[["unit", "t"]].merge(labels[["unit", "t", "y_240"]], on=["unit", "t"], how="left")
    return LT.table(LT.lifespans_of(g, scores, events, buds), thresholds)


def det_keys(labels, p: str, h: int = 240):
    return labels[(labels["position"] == p) & labels[f"y_{h}"].notna()][["unit", "t"]].reset_index(drop=True)


def average_tables(tables: list[dict]) -> dict:
    """The row's per-model lead-time figures averaged per position (the a_R(p) convention)."""
    out = {}
    for lvl in ("a=0.05", "time_shuffle_a=0.05"):
        rows = [t[lvl] for t in tables]
        avg = {}
        for k in ("median_lead_min", "n_never_flagged", "share_at_cap", "false_alarm_rate",
                  "alarms_per_100_negative_hours", "share_eventfree_lifespans_ever_alarm",
                  "median_buds_onset_to_death"):
            v = [r[k] for r in rows if r.get(k) is not None]
            avg[k] = float(np.mean(v)) if v else None
        avg["share_flagged_by"] = {h: float(np.mean([r["share_flagged_by"][h] for r in rows]))
                                   for h in rows[0]["share_flagged_by"]} if rows else None
        out[lvl] = avg
    return out


def _bar_lead(bar_lt: dict, p: str) -> float:
    """The best rule's median lead time at a = 0.05 on p, 0 when not computable."""
    best = bar_lt[p].get("best") or {}
    return (best.get("a=0.05") or {}).get("median_lead_min") or 0


def bar_lead_time(rows, pic, p: str, bar_240: dict, labels, events: dict, buds: dict) -> dict:
    """Each rule refitted on the 5 threshold-run positions and thresholded on the threshold position's negative
    windows, then (as fitted on all 6, ``bar_240``) applied to the held-out position. The rules' lead time is the
    best rule's (the longest median lead time at a = 0.05)."""
    from . import rules as B
    from . import leadtime as LT
    thr_pos = C.threshold_position(p)
    fit5 = B.bar_for_position(rows, pic, thr_pos, horizons=(240,), train_positions=C.threshold_run_positions(p))[240]
    y5 = fit5["keys"].merge(labels[["unit", "t", "y_240"]], on=["unit", "t"], how="left")["y_240"].to_numpy()
    members = {}
    for m, s_thr in fit5["scores"].items():
        if m == "single_best_on_heldout" or m not in bar_240["scores"]:
            continue
        th = LT.thresholds_from(s_thr[y5 == 0])
        members[m] = lead_table_on(labels, bar_240["keys"], bar_240["scores"][m], th, events, buds)
    if not members:
        return {"members": {}, "best_member": None, "best": None, "threshold_position": thr_pos,
                "not_computable": "no bar member is scored at 240 min on both the threshold and held-out positions"}
    best = max(members, key=lambda m: (members[m]["a=0.05"]["median_lead_min"] or 0.0))
    g = bar_240["keys"][["unit", "t"]].merge(labels[["unit", "t", "y_240"]], on=["unit", "t"], how="left")
    ci = LT.median_lead_bootstrap([LT.lifespans_of(g, bar_240["scores"][best], events, buds)],
                                  [members[best]["a=0.05"]["threshold"]])
    return {"members": members, "best_member": best, "best": members[best], "threshold_position": thr_pos,
            "best_median_lead_95ci": ci,
            "note": "the best rule on the held-out position (the conservative choice), kept fixed in the "
                    "bootstrap"}


def surv_auprc(labels, p: str, frame, col: str, h: int) -> float:
    from .metrics import auprc
    g = labels[(labels["position"] == p) & labels[f"y_{h}"].notna()]
    m = g.merge(frame[["unit", "t", col]], on=["unit", "t"], how="left", validate="one_to_one")
    if m[col].isna().any():
        raise ValueError(f"the survival layer has no score for a row of {p} at {h} min")
    return float(auprc(m[f"y_{h}"].astype(int).to_numpy(), m[col].to_numpy()))


def next_bud_print(rows, ok, surv_file) -> dict:
    """Per start and position, the mean over seeds of the AUPRC of the next-bud incidence against 'a bud within h'
    and of the death incidence against 'death within h' (h = 30 / 60 / 120 / 240 min), each on the landmarks whose
    status at h is determined for that fate; from the two-cause survival layer. ``surv_file(run, kind)`` locates a
    survival-layer frame."""
    from .metrics import auprc
    out = {}
    for st in C.STARTS:
        per = {}
        for p in C.POSITIONS:
            g = rows[rows["position"] == p]
            res = {}
            for h in C.HORIZONS:
                k = h // C.FRAME_MIN
                nb, ev, ob = g["nb_off"].to_numpy(float), g["ev_off"].to_numpy(float), g["obs_off"].to_numpy(float)
                bud_y = np.where(np.isfinite(nb) & (nb <= k) & ~(np.isfinite(ev) & (ev < nb)), 1.0,
                                 np.where((np.isfinite(nb) & (nb > k)) | (np.isfinite(ev) & (ev <= k)) | (ob >= k),
                                          0.0, np.nan))
                vals = {"bud": [], "death": []}
                for s_ in C.SEEDS:
                    u = run_name(st, p, s_)
                    if not ok(u):
                        continue
                    fr = g[["unit", "t", f"y_{h}"]].merge(pd.read_parquet(surv_file(u, "NB")), on=["unit", "t"])
                    yb = fr[["unit", "t"]].merge(g[["unit", "t"]].assign(yb=bud_y), on=["unit", "t"])["yb"].to_numpy()
                    mb = np.isfinite(yb)
                    if mb.any() and yb[mb].sum():
                        vals["bud"].append(float(auprc(yb[mb].astype(int), fr[f"cif_bud_{h}"].to_numpy()[mb])))
                    md = fr[f"y_{h}"].notna().to_numpy()
                    if md.any() and fr.loc[md, f"y_{h}"].sum():
                        vals["death"].append(float(auprc(fr.loc[md, f"y_{h}"].astype(int).to_numpy(),
                                                         fr.loc[md, f"cif_death_{h}"].to_numpy())))
                res[str(h)] = {f: (float(np.mean(v)) if v else None) for f, v in vals.items()}
            per[p] = res
        out[st] = per
    return out


def survival_layer_scores(rec: dict, rows, series, labels, bar: dict, bar_lt: dict, sc: dict, ok, surv_file,
                          events: dict, buds: dict) -> None:
    """The sections read from the survival layer (the survival head on each run's own embeddings): the head's AUPRC
    at every horizon (embeddings alone, S, and with the label features, SL), its lead time with the threshold from
    the fold's threshold run, the threshold-free survival scores and the next-bud split. Adds them to ``rec``."""
    from . import leadtime as LT
    from . import survival_scores as SS
    sec, lt = {}, {}
    for st in C.STARTS:
        per_h = {}
        for h in C.HORIZONS:
            vals = {}
            for p in C.POSITIONS:
                v = {row: [surv_auprc(labels, p, pd.read_parquet(surv_file(run_name(st, p, s), row)),
                                      f"cif_{h}", h) for s in C.SEEDS if ok(run_name(st, p, s))]
                     for row in ("S", "SL")}
                vals[p] = {row: (sum(x) / len(x) if x else None) for row, x in v.items()}
            entry = {"per_position": vals}
            if h != C.PRIMARY_H and sc[st]["scorable"]:
                entry["reading_S_vs_bar_secondary"] = classify(
                    {p: vals[p]["S"] - bar[p][h]["bar"]["auprc"] for p in C.POSITIONS}, "bar")
            per_h[str(h)] = entry
        sec[st] = per_h
        lt_st = {}
        for p in C.POSITIONS:
            tu = f"thr__{st}__{p}"
            if not surv_file(tu, "S").is_file():
                lt_st[p] = "not computable: no threshold run"
                continue
            tp = C.threshold_position(p)
            sv = pd.read_parquet(surv_file(tu, "S"))
            g = det_keys(labels, tp).merge(labels[["unit", "t", "y_240"]], on=["unit", "t"])
            neg = aligned_scores(g[g["y_240"] == 0], [sv.rename(columns={"cif_240": "p_death"})])[0]
            th = LT.thresholds_from(neg)
            tabs = {}
            for s_ in C.SEEDS:
                u = run_name(st, p, s_)
                if ok(u):
                    fr = pd.read_parquet(surv_file(u, "S")).rename(columns={"cif_240": "p_death"})
                    dk = det_keys(labels, p)
                    tabs[u] = lead_table_on(labels, dk, aligned_scores(dk, [fr])[0], th, events, buds)
            dk = det_keys(labels, p)
            g_lab = dk.merge(labels[["unit", "t", "y_240"]], on=["unit", "t"], how="left")
            lss = [LT.lifespans_of(g_lab, aligned_scores(dk, [pd.read_parquet(surv_file(u, "S")).rename(
                columns={"cif_240": "p_death"})])[0], events, buds) for u in tabs]
            lt_st[p] = {"thresholds": {str(k): v for k, v in th.items()}, "threshold_position": tp,
                        "per_model": tabs, "average": average_tables(list(tabs.values())) if tabs else None,
                        "median_lead_95ci": LT.median_lead_bootstrap(lss, [th[LT.A_MAIN]] * len(lss))}
        lt_st["k_of_7_longer_than_bar"] = sum(
            isinstance(lt_st.get(p), dict) and lt_st[p]["average"] is not None
            and (lt_st[p]["average"]["a=0.05"]["median_lead_min"] or 0)
            > _bar_lead(bar_lt, p) for p in C.POSITIONS)
        lt[st] = lt_st
    rec["secondary_horizons"] = sec
    n_frames = D.lifespan_table(series).set_index("unit")["n_frames"].to_dict()
    model_frames = {p: {st: {s_: pd.read_parquet(surv_file(run_name(st, p, s_), "S")) for s_ in C.SEEDS
                             if ok(run_name(st, p, s_))} for st in C.STARTS} for p in C.POSITIONS}
    rec["survival_scores"] = SS.survival_layer_scores(rows, n_frames, model_frames,
                                                      {p: bar[p][C.PRIMARY_H].get("surv_full") for p in C.POSITIONS})
    rec["next_bud_vs_death"] = next_bud_print(rows, ok, surv_file)
    rec["lead_time"] = {"per_start": lt, "score": "the survival layer's P(death within 240 min)"}


# ------------------------------------------------------------------ the bundle
def run_name(start: str, position: str, seed: int) -> str:
    return f"{start}__{position}__s{seed}"


def score_bundle(bundle: Path, *, with_sensitivity: bool = True, with_lead_time: bool = True,
                 verify: bool = True) -> dict:
    """Everything the stored predictions determine: the rules per position, per-run AUPRC, a_R per start, the
    readings against the best rule and between the starts, intervals, spreads, AUROC, the label-efficiency rows,
    the phase split and the label sensitivity. ``verify`` checks the files against the bundle's MANIFEST.json first."""
    from . import rules as B
    bundle = Path(bundle)
    if verify:
        from deadahead.reproduce.bundle import check_files, read_manifest
        bad = check_files(bundle, read_manifest(bundle))
        if bad:
            raise ValueError("the bundle does not match its manifest: " + "; ".join(bad[:5]))
    series, rows, _, pics = D.load_tables(bundle / "labels")
    bar = {p: B.bar_for_position(rows, pics[p], p) for p in C.POSITIONS}
    no_event = [p for p in C.POSITIONS if "bar" not in bar[p][C.PRIMARY_H]]
    if no_event:
        raise ValueError(f"no rule is scorable at {C.PRIMARY_H} min on {no_event}: no death in the position or its "
                         "training positions")
    labels = rows[["unit", "position", "t", *[f"y_{h}" for h in C.HORIZONS]]]
    pred_dir = bundle / "predictions"

    def ok(u):
        return (pred_dir / f"{u}.parquet").is_file()

    def read(u, p):
        pr = pd.read_parquet(pred_dir / f"{u}.parquet")
        pr.attrs["positions"] = (p,)
        return pr

    rec = {"bar": {}}
    for p in C.POSITIONS:
        rec["bar"][p] = B.summary(bar[p][C.PRIMARY_H])
        rec["bar"][p]["auprc"] = bar[p][C.PRIMARY_H]["bar"]["auprc"]
        rec["bar"][p]["member"] = bar[p][C.PRIMARY_H]["bar"]["member"]
    bar_p = {p: rec["bar"][p]["auprc"] for p in C.POSITIONS}
    rec["bar_secondary_horizons"] = {str(h): {p: {"auprc": bar[p][h]["bar"]["auprc"], "member": bar[p][h]["bar"]["member"]}
                                              for p in C.POSITIONS} for h in C.HORIZONS[1:]}
    sc = {st: survival_layer_scorable(st, ok) for st in C.STARTS}
    units = {st: [run_name(st, p, s) for p in C.POSITIONS for s in C.SEEDS if ok(run_name(st, p, s))]
             for st in C.STARTS}
    preds = {u: read(u, u.split("__")[1]) for st in C.STARTS for u in units[st]}
    for u, pr in preds.items():
        pr.attrs["model"] = u.rsplit("__", 1)[1]                    # a seed's column joins that seed
    per_all = per_position_auprc(labels, preds)
    rows_by_start = {st: row_means(per_all, units[st]) for st in C.STARTS}
    rec.update({"scorable": sc, "per_model": per_all, "a_R": rows_by_start})
    frames = {st: {p: [preds[u] for u in units[st] if u.split("__")[1] == p] for p in C.POSITIONS} for st in C.STARTS}
    rec["intervals"] = {p: position_intervals(bar[p][C.PRIMARY_H], labels, {st: frames[st][p] for st in C.STARTS})
                        for p in C.POSITIONS}
    readings = {st: classify({p: rows_by_start[st][p] - bar_p[p] for p in C.POSITIONS}, "bar")
                for st in C.STARTS if sc[st]["scorable"]}
    rec["comparison_A"] = readings.get(C.STARTS[0], "not scorable")
    rec["comparison_A2_secondary"] = readings.get(C.STARTS[1], "not scorable")
    if all(sc[st]["scorable"] for st in C.STARTS):
        rec["comparison_B"] = classify({p: rows_by_start[C.STARTS[0]][p] - rows_by_start[C.STARTS[1]][p]
                                        for p in C.POSITIONS}, "transfer")
    else:
        rec["comparison_B"] = "not scorable: needs both starts"
    le = {}
    for st in C.STARTS:
        for p in C.POSITIONS:
            u = f"le__{st}__{p}"
            if ok(u):
                six = per_all.get(run_name(st, p, C.SEED), {}).get(p)
                le[u] = {"auprc_3_positions": per_position_auprc(labels, {u: read(u, p)})[u][p],
                         "auprc_6_positions_seed2601562": six,
                         "positions": list(D.label_efficiency_positions(p)),
                         "training_deaths_3": int(sum(C.DEATHS_BY_POSITION[q]
                                                      for q in D.label_efficiency_positions(p))),
                         "training_deaths_6": int(sum(C.DEATHS_BY_POSITION[q] for q in C.training_positions(p)))}
    rec["label_efficiency"] = le
    rec["pos16_7_row"] = {st: rows_by_start[st].get("Pos16_7") for st in C.STARTS}
    rec["pos16_7_row"]["bar"] = bar_p["Pos16_7"]
    rec["spread_over_seeds"] = {st: {p: spread(per_all, units[st], p) for p in C.POSITIONS} for st in C.STARTS}
    rec["auroc"] = per_position_auroc(labels, preds)
    rec["phase_split"] = split_by_phase(rows, bar, frames)
    events, buds = lifespan_facts(series)
    if with_lead_time:
        bar_lt = {p: bar_lead_time(rows, pics[p], p, bar[p][240], labels, events, buds) for p in C.POSITIONS}
        rec["bar_lead_time"] = bar_lt
    surv_dir = bundle / "survival"
    if with_lead_time and surv_dir.is_dir():
        survival_layer_scores(rec, rows, series, labels, bar, bar_lt, sc, ok,
                              lambda u, row: surv_dir / f"{u}__{row}.parquet", events, buds)
    if with_sensitivity:
        rec["sensitivity"] = sensitivity_rows(series, pics, rows, {st: frames[st] for st in C.STARTS
                                                                   if sc[st]["scorable"]},
                                              {st: readings[st]["class"] for st in readings})
    rec["summary"] = {
        "n_positions": len(C.POSITIONS),
        "macro_auprc_30min": {st: float(np.mean([rows_by_start[st][p] for p in C.POSITIONS])) for st in C.STARTS
                              if len(rows_by_start[st]) == len(C.POSITIONS)},
        "macro_auprc_30min_best_rule": float(np.mean(list(bar_p.values()))),
        "ahead_of_best_rule": {st: readings[st]["k_plus"] for st in readings},
        "finetuned_ahead_of_public_start": (rec["comparison_B"]["k_plus"] if isinstance(rec["comparison_B"], dict)
                                            else None)}
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=None, help="the yeast bundle (default: <DEADAHEAD_DATA>/yeast_bundle)")
    ap.add_argument("--out", type=Path, default=None, help="write the scores as json")
    ap.add_argument("--no-sensitivity", action="store_true", help="skip the label sensitivity (refits the rules)")
    ap.add_argument("--no-lead-time", action="store_true",
                    help="skip the lead times and the survival-layer sections (refits the rules on 5 positions)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the size and sha256 check of the bundle files (your own runs have no MANIFEST.json)")
    a = ap.parse_args(argv)
    if a.out is not None and not a.out.parent.is_dir():
        ap.error(f"--out: no folder {a.out.parent}")
    bundle = a.bundle or default_bundle()
    if D.tables_missing(bundle / "labels"):
        print(f"no yeast bundle at {bundle}: unzip deadahead_yeast_bundle_v1.zip into $DEADAHEAD_DATA "
              "(docs/WEIGHTS.md)", file=sys.stderr)
        return 3
    if not a.no_verify and not (bundle / "MANIFEST.json").is_file():
        print(f"no MANIFEST.json in {bundle}: pass --no-verify for your own runs", file=sys.stderr)
        return 3
    try:
        rec = score_bundle(bundle, with_sensitivity=not a.no_sensitivity, with_lead_time=not a.no_lead_time,
                           verify=not a.no_verify)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    if a.out is not None:
        a.out.write_text(json.dumps(rec, indent=1, default=str) + "\n")
        print(f"wrote {a.out}")
    s = rec["summary"]
    print(f"macro AUPRC at 30 min over {s['n_positions']} held-out positions:")
    for st in C.STARTS:
        v, k = s["macro_auprc_30min"].get(st), s["ahead_of_best_rule"].get(st)
        macro = "not scorable" if v is None else f"{v:.4f}"
        ahead = "not scorable" if k is None else f"ahead of the best rule on {k} of 7"
        print(f"  {st:<20} {macro}   {ahead}")
    print(f"  {'best simple rule':<20} {s['macro_auprc_30min_best_rule']:.4f}   (picked on each test position)")
    k = s["finetuned_ahead_of_public_start"]
    print("fine-tuned start against the public start: " + ("not scorable" if k is None else f"ahead on {k} of 7"))
    for key, name in COMPARISON_NAMES.items():
        r = rec[key]
        if isinstance(r, dict):
            print(f"{name}: {r['class']} (mean Delta {r['mean_delta']:+.4f}, {r['k_plus']} of 7 ahead)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
