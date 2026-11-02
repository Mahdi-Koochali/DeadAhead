"""The simple rules on each held-out position, the best of them, and the one-cause survival head. CPU only.

On each held-out position p and horizon, the best simple rule is the highest AUPRC on p of:
  1. ``label_combined``   one LightGBM on all the label features;
  2. ``single:<feature>`` every single label feature, oriented and given its missing side on the training
                          positions; the best of them on p;
  3. ``label_surv``       the label features through the survival head;
  4. ``pic_bright_lgb``, ``pic_bright_surv``, ``pic_size_lgb``, ``pic_size_surv``: whole-trap brightness statistics
                          alone and with the size proxy, each as one LightGBM and through the survival head;
  5. ``label_pic_surv``   the label features and the picture measurements together through the survival head.
Picking the rule on p favours the rules. Printed beside them: the label-only value (the higher of rule 1 and the
single feature best on the training positions), a frame-class reference and an absolute-frame-index shortcut.

The survival head is a discrete-time cause-specific hazard model (person-period expansion, multiclass LightGBM with
the bin index as a feature): bins of 6 frames, 8 bins, 200 rounds, weight 1. In names and result keys, ``bar`` is the
best simple rule.

    deadahead yeast-rules [--horizons 30 60 120 240]       rule AUPRCs per held-out position (rules.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from deadahead.survival.head import usable_cpus

from . import constants as C
from . import lifespans as L
from .data import PIC_BRIGHT, PIC_SIZE, survival_bins

N_BINS = 8                                  # 8 bins of 30 min (240 min)
BIN_OF_H = {30: 1, 60: 2, 120: 4, 240: 8}
SURV_ROUNDS = 200
BAR_SURV_SEED = C.SEED                      # the survival fits' seed; the LightGBM rules keep seed 0
_N_THREADS = int(os.environ.get("OMP_NUM_THREADS") or usable_cpus())
#: Multiclass LightGBM settings of the survival head (num_class is set per call: causes + 1).
LGB_SURV = {"objective": "multiclass", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
            "num_threads": _N_THREADS, "deterministic": True, "force_col_wise": True}
LABEL_FEATURES = L.DETECDIV_FEATURES
PIC_MEMBERS = {"pic_bright": PIC_BRIGHT, "pic_size": PIC_SIZE}
BAR_MEMBER_FITS = ("label_combined", "single_best_on_heldout", "label_surv", "pic_bright_lgb", "pic_bright_surv",
                   "pic_size_lgb", "pic_size_surv", "label_pic_surv")


# ------------------------------------------------------------------ the survival head
def survival_cif(Xtr: np.ndarray, ev_bin: np.ndarray, cause: np.ndarray, cens_bin: np.ndarray, Xte: np.ndarray, *,
                 seed: int, n_causes: int = 1, n_bins: int = N_BINS, rounds: int = SURV_ROUNDS,
                 params: dict | None = None) -> dict | None:
    """Discrete-time cause-specific hazards on a person-period expansion; returns {bin k: (n_te, n_causes)
    cumulative incidence}, monotone in k. None when the expansion holds one class only."""
    import lightgbm as lgb
    Xb = np.asarray(Xtr, dtype=np.float32)
    ev_bin = np.asarray(ev_bin, dtype=float)
    cens_bin = np.asarray(cens_bin, dtype=float)
    cause = np.asarray(cause, dtype=int)
    X, Y, W = [], [], []
    for k in range(1, n_bins + 1):
        at_risk = (ev_bin >= k) & (cens_bin >= k)
        if not at_risk.any():
            continue
        Y.append(np.where(ev_bin[at_risk] == k, cause[at_risk], 0))
        X.append(np.column_stack([Xb[at_risk], np.full(at_risk.sum(), k, dtype=np.float32)]))
        W.append(np.ones(int(at_risk.sum())))
    if not X:
        return None
    Xp, Yp, Wp = np.vstack(X), np.concatenate(Y), np.concatenate(W)
    if len(np.unique(Yp)) < 2:
        return None
    prm = {**(params if params is not None else LGB_SURV), "num_class": n_causes + 1, "seed": seed}
    m = lgb.train(prm, lgb.Dataset(Xp, label=Yp, weight=Wp), num_boost_round=rounds)
    Xt = np.asarray(Xte, dtype=np.float32)
    n = len(Xt)
    surv, cum, out = np.ones(n), np.zeros((n, n_causes)), {}
    for k in range(1, n_bins + 1):
        p = m.predict(np.column_stack([Xt, np.full(n, k, dtype=np.float32)]))
        cum = cum + surv[:, None] * p[:, 1:]
        surv = surv * p[:, 0]
        out[k] = cum.copy()
    return out


def survival_member(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str], *, seed: int) -> dict | None:
    """The one-cause head fitted on every training landmark (censored included, never a negative after censoring);
    the death incidence of the test rows at each horizon: {h: array}."""
    ev_bin, cens_bin = survival_bins(tr)
    cause = np.isfinite(ev_bin).astype(int)
    out = survival_cif(tr[cols].to_numpy(float), ev_bin, cause, cens_bin, te[cols].to_numpy(float), seed=seed)
    if out is None:
        return None
    return {h: out[BIN_OF_H[h]][:, 0] for h in C.HORIZONS}


# ------------------------------------------------------------------ the rules
def _lgb(tr: pd.DataFrame, te: pd.DataFrame, cols, y: str, categorical=()) -> np.ndarray:
    """One LightGBM per horizon with the fixed label-rule settings and seed."""
    import lightgbm as lgb
    m = lgb.train({**L.LGB_PARAMS, "seed": L.LGB_SEED},
                  lgb.Dataset(tr[list(cols)], label=tr[y].to_numpy(), categorical_feature=list(categorical) or "auto"),
                  num_boost_round=L.NUM_ROUNDS)
    return m.predict(te[list(cols)])


def bar_for_position(rows: pd.DataFrame, pic: pd.DataFrame, heldout: str, *, horizons=C.HORIZONS,
                     surv_seed: int = BAR_SURV_SEED, train_positions: tuple | None = None) -> dict:
    """Every rule's scores and AUPRC on the held-out position's determined rows at each horizon, the best rule (the
    maximum, with the rule that set it) and the printed rows. ``pic`` holds the fold's picture measurements, aligned
    to ``rows`` by index. ``train_positions`` defaults to the other 6 positions."""
    from .metrics import auprc, auroc
    if not rows.index.equals(pic.index):
        raise ValueError("the picture measurements are not aligned to the rows")
    d = pd.concat([rows, pic[list(PIC_SIZE)]], axis=1)
    train = tuple(train_positions) if train_positions is not None else tuple(q for q in C.POSITIONS if q != heldout)
    tr_all, te_all = d[d["position"].isin(train)], d[d["position"] == heldout]
    if heldout in set(tr_all["position"]) or len(te_all) == 0:
        raise ValueError(f"fold {heldout}: the training rows hold the held-out position, or it has no rows")
    surv = {"label_surv": survival_member(tr_all, te_all, list(LABEL_FEATURES), seed=surv_seed),
            "pic_bright_surv": survival_member(tr_all, te_all, list(PIC_BRIGHT), seed=surv_seed),
            "pic_size_surv": survival_member(tr_all, te_all, list(PIC_SIZE), seed=surv_seed),
            "label_pic_surv": survival_member(tr_all, te_all, list(LABEL_FEATURES) + list(PIC_SIZE), seed=surv_seed)}
    out = {}
    for h in horizons:
        y = f"y_{h}"
        tr, te_mask = tr_all[tr_all[y].notna()].copy(), te_all[y].notna().to_numpy()
        te = te_all[te_mask].copy()
        tr["yy"], te["yy"] = tr[y].astype(int), te[y].astype(int)
        yte, ytr = te["yy"].to_numpy(), tr["yy"].to_numpy()
        r = {"n_landmarks": int(len(te)), "n_positive": int(yte.sum()), "event_rate": float(yte.mean()),
             "keys": te[["unit", "t"]].reset_index(drop=True), "members": {}, "scores": {},
             # the survival rules' incidence on all of the position's rows
             "surv_full": {"keys": te_all[["unit", "t"]].reset_index(drop=True), "cif": surv}}
        if yte.sum() == 0 or ytr.sum() == 0:
            out[h] = {**r, "skipped": "no event in the held-out or the training positions"}
            continue

        def add(name, s):
            r["scores"][name] = np.asarray(s, dtype=float)
            r["members"][name] = {"auprc": float(auprc(yte, s)), "auroc": float(auroc(yte, s))}
        add("label_combined", _lgb(tr, te, LABEL_FEATURES, "yy"))
        singles, best_tr = {}, None
        for x in LABEL_FEATURES:
            s_tr, s_te, info = L.single_scores(tr[x].to_numpy(float), ytr, te[x].to_numpy(float))
            if s_tr is None:
                singles[x] = info
                continue
            a_tr = float(auprc(ytr, s_tr))
            singles[x] = {**info, "train_auprc": a_tr, "auprc": float(auprc(yte, s_te))}
            r["scores"][f"single:{x}"] = s_te
            if best_tr is None or a_tr > best_tr[1]:
                best_tr = (x, a_tr)
        scored = {x: v for x, v in singles.items() if "auprc" in v}
        best_te = max(scored, key=lambda x: (scored[x]["auprc"], -LABEL_FEATURES.index(x)))
        r["singles"] = singles
        r["members"]["single_best_on_heldout"] = {"feature": best_te, "auprc": scored[best_te]["auprc"]}
        r["scores"]["single_best_on_heldout"] = r["scores"][f"single:{best_te}"]
        for name, cols in PIC_MEMBERS.items():
            add(f"{name}_lgb", _lgb(tr, te, cols, "yy"))
        for name, s in surv.items():
            if s is None:
                r["members"][name] = {"skipped": "the survival expansion holds one class only"}
            else:
                add(name, s[h][te_mask])
        fits = {k: v for k, v in r["members"].items() if "auprc" in v}
        top = max(fits, key=lambda k: (fits[k]["auprc"], -BAR_MEMBER_FITS.index(k)))
        r["bar"] = {"auprc": fits[top]["auprc"], "member": top if top != "single_best_on_heldout"
                    else f"single:{best_te}", "member_key": top,
                    "note": "picked on the held-out position, which favours the rules"}
        comb = r["members"]["label_combined"]["auprc"]
        label_single = singles[best_tr[0]]["auprc"] if best_tr else None
        r["label_only_value"] = {"auprc": max(comb, label_single) if label_single is not None else comb,
                             "best_single_on_training": best_tr[0] if best_tr else None,
                             "combined_auprc": comb, "note": "the label-only value: printed, never the best rule"}
        ref = _lgb(tr, te, tuple(LABEL_FEATURES) + ("frame_class",), "yy", categorical=("frame_class",))
        r["reference_frame_class"] = {"auprc": float(auprc(yte, ref)), "note": "a reference, not a rule"}
        s_tr, s_te, info = L.single_scores(tr["t"].to_numpy(float), ytr, te["t"].to_numpy(float))
        r["shortcut_frame_index"] = {"auprc": float(auprc(yte, s_te)), **info,
                                     "note": "absolute frame index only (time in the movie): a shortcut check"}
        out[h] = r
    return out


# ------------------------------------------------------------------ intervals
def bootstrap(y: np.ndarray, model_scores: dict, bar_scores: np.ndarray, clusters: np.ndarray, *,
              n_boot: int = 2000, seed: int = C.SEED) -> dict:
    """Inside one held-out position, resample whole lifespans; per resample, a_R = the mean AUPRC over each row's
    models, the best rule picked on the full position kept fixed, Delta = a_R - best rule. 95% percentile
    intervals. ``model_scores``: {row: [score arrays, one per model]}."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(pd.Series(clusters))
    k = len(uniq)
    y = np.asarray(y, dtype=float)
    prep_bar = L._ap_prepare(y, np.asarray(bar_scores, dtype=float))
    preps = {r: [L._ap_prepare(y, np.asarray(s, dtype=float)) for s in v] for r, v in model_scores.items()}
    draws = {"bar": np.full(n_boot, np.nan), **{r: np.full(n_boot, np.nan) for r in preps},
             **{f"delta:{r}": np.full(n_boot, np.nan) for r in preps}}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k)[codes].astype(float)
        if (y * w).sum() == 0:
            continue
        draws["bar"][b] = L._weighted_ap(prep_bar, w)
        for r, pl in preps.items():
            draws[r][b] = float(np.mean([L._weighted_ap(pp, w) for pp in pl]))
            draws[f"delta:{r}"][b] = draws[r][b] - draws["bar"][b]

    def ci(v):
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    return {"n_resamples": n_boot, "seed": seed, "n_lifespans": int(k), **{key: ci(v) for key, v in draws.items()}}


# ------------------------------------------------------------------ inputs and CLI
def summary(bar_h: dict) -> dict:
    """The printed fields of one position and horizon (no score arrays)."""
    return {k: bar_h[k] for k in ("bar", "members", "label_only_value", "reference_frame_class", "shortcut_frame_index",
                                  "event_rate", "n_landmarks", "n_positive") if k in bar_h}


def main(argv=None) -> int:
    from .data import load_tables, tables_missing, yeast_dir
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=Path, default=None,
                    help="folder with classes.parquet, brightness.parquet, size.parquet and folds.json "
                         "(default: <DEADAHEAD_DATA>/yeast; the bundle's labels/ folder also works)")
    ap.add_argument("--horizons", type=int, nargs="+", choices=list(C.HORIZONS), default=list(C.HORIZONS))
    ap.add_argument("--out", type=Path, default=None, help="output json (default: <tables>/rules.json)")
    a = ap.parse_args(argv)
    if a.out is not None and not a.out.parent.is_dir():
        ap.error(f"--out: no folder {a.out.parent}")
    tdir = a.tables or yeast_dir()
    missing = tables_missing(tdir)
    if missing:
        print(f"{tdir} holds no {', '.join(missing)}: run deadahead yeast-data, or pass --tables <yeast bundle>/labels",
              file=sys.stderr)
        return 3
    _, rows, _, pics = load_tables(tdir)
    res = {}
    for p in C.POSITIONS:
        bar = bar_for_position(rows, pics[p], p, horizons=tuple(a.horizons))
        res[p] = {str(h): summary(bar[h]) for h in a.horizons}
        best = res[p].get(str(a.horizons[0]), {}).get("bar", {})
        print(f"{p}: best rule at {a.horizons[0]} min {best.get('member')} AUPRC {best.get('auprc')}")
    out = a.out or (Path(tdir) / "rules.json")
    out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
