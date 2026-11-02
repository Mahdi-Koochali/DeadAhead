"""Train the DeadAhead head for one held-out group and seed, and write its predictions and booster.

    deadahead train-head --group DU145_bp --seed 2601562
    deadahead train-head --group DU145_bp --seed 2601562 \\
        --booster deadahead_head__DU145_bp__seed2601562.txt    (predict only)
    deadahead train-head --group DU145_bp --seed 2601562 --inner-group PNT1A_do

The head is fitted on the other eight groups with the training labels of the truncated movies, and predicts the
cumulative incidence of apoptosis and lytic death at 30, 60, 120 and 240 min for every landmark of the held-out group.
``--fold`` is accepted for ``--group``, as in the video commands. The DeadAhead booster is written as
``deadahead_head__<group>__seed<seed>.txt``, the name of the released boosters.

``--inner-group H`` fits a cross-fitted inner head instead: the same head on the seven training groups other than the
held-out group and H, reading the features of the model that held out ``--group`` (same seed), and predicts H. The
eight inner heads of a (group, seed) give the training-only alarm thresholds of the warnings. Their files carry
``__h<H>`` in the name.

``--method videomae_adapted_survival`` reads the features of the adapted encoder through ``--features``:
``deadahead video-features --fold <group> --adapted`` writes them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from deadahead.features.history import history_features
from deadahead.paths import runs_dir
from deadahead.survival import head
from deadahead.survival.boosters import (
    base_matrix,
    capture_booster,
    load_booster,
    max_abs_diff,
    predict_incidence,
    save_booster,
)
from deadahead.survival.config import head_config, windows_config
from deadahead.survival.frames import (
    CUT_FRAME,
    GROUPS,
    METHODS,
    PANELS,
    apply_view,
    assert_inside_cut,
    build_data,
    coherence_violations,
    inner_split,
    input_columns,
    join_store,
    landmark_table,
    load_tables,
    panel_view,
    panel_windows,
    prediction_payload,
    split_fold,
    store_positions,
)

ROUNDTRIP_TOL = 1e-9           # a saved booster must give the fitted head's incidences to this absolute tolerance
VIDEO_PREFIX = "videomae_"


def default_features(group: str, seed: int) -> Path:
    """Fine-tuned VideoMAE features of the model that held out ``group`` with ``seed``: (n_rows, 1024) float32."""
    return runs_dir() / "videomae" / "features" / f"{group}__s{seed}.npy"


def default_rows() -> Path:
    """The landmark keys of the feature arrays, one row per array row."""
    return runs_dir() / "videomae" / "features" / "rows.parquet"


def run_tag(untyped: str = "censor_at_death", inner_group: str | None = None) -> str:
    """What a run adds to its file names: the rule for untyped deaths, unless it is the paper's, and the inner group."""
    rule = "" if untyped == "censor_at_death" else f"__{untyped}"
    return rule if inner_group is None else f"{rule}__h{inner_group}"


def booster_name(method: str, group: str, seed: int, panel: str, tag: str = "") -> str:
    """File name of a fitted booster: the released name for DeadAhead on the primary panel, else method, group,
    seed and panel; ``tag`` comes from ``run_tag``."""
    if method == "deadahead":
        stem = f"deadahead_head__{group}__seed{seed}"
        return f"{stem}{tag}.txt" if panel == "primary" else f"{stem}__{panel}{tag}.txt"
    return f"{method}__{group}__s{seed}__{panel}{tag}.txt"


def load_features(path: Path, n_rows: int) -> np.ndarray:
    a = np.load(Path(path), allow_pickle=False)
    if a.ndim != 2 or a.shape[0] != n_rows or a.dtype != np.float32:
        raise ValueError(f"{path}: shape {a.shape} {a.dtype}; ({n_rows}, 1024) float32 expected")
    if not np.isfinite(a).all():
        raise ValueError(f"{path}: a non-finite value")
    return a


def fit_head(group: str, seed: int, tables: dict, hist: pd.DataFrame, cfg: dict, cfg_w: dict, *,
             panel: str = "primary", method: str = "deadahead", rows: pd.DataFrame | None = None,
             feats: np.ndarray | None = None, untyped: str = "censor_at_death", inner_group: str | None = None):
    """Fit one head and predict the held-out group, or with ``inner_group`` fit on the seven groups other than
    ``group`` and ``inner_group`` and predict ``inner_group``. ``feats`` are the features of the model that held out
    ``group`` in both cases.

    Returns (incidence dict of ``train_predict_survival``, booster, predicted rows, their input matrix).
    """
    data, phys = build_data(tables["windows"], tables["events"], tables["movies"], hist, cfg, cfg_w, panel)
    lm = landmark_table(panel_windows(tables["windows"], panel))
    tr0, te = split_fold(data, group)
    if feats is not None:
        store_positions(data, rows)                      # every landmark has a feature row
    view = panel_view(lm, untyped, tables["movies"], CUT_FRAME)
    tr = apply_view(tr0, view)
    assert_inside_cut(tr, tables["movies"])
    if inner_group is not None:
        tr, te = inner_split(tr, data, group, inner_group)
    del tr0, data
    te = te.assign(censor_offset=np.inf)                 # the held-out side is never censored
    trx, tex, vcols = tr, te, []
    if feats is not None:
        trx, vcols = join_store(tr, rows, feats, VIDEO_PREFIX)
        tex, _ = join_store(te, rows, feats, VIDEO_PREFIX)
    cols = input_columns(method, vcols, phys)
    out, boosters = capture_booster(head.train_predict_survival, trx, tex, cols, cfg, cfg_w, int(seed),
                                    interval_censored=False)
    if not out or "_index" not in out or len(boosters) != 1:
        raise RuntimeError("no head was fitted (no row at risk, or a single class)")
    del trx
    keys, Xt = base_matrix(tex, cols)
    if not keys.equals(out["_index"]):
        raise AssertionError("the held-out matrix rows differ from the fitted head's")
    d = max_abs_diff(predict_incidence(boosters[0], Xt, cfg, cfg_w), out)
    if not d <= ROUNDTRIP_TOL:
        raise AssertionError(f"the captured booster does not reproduce the head's incidences ({d})")
    return out, boosters[0], te, Xt


def predict_heldout(group: str, booster, tables: dict, hist: pd.DataFrame, cfg: dict, cfg_w: dict, *,
                    panel: str = "primary", method: str = "deadahead", rows: pd.DataFrame | None = None,
                    feats: np.ndarray | None = None):
    """Predict the held-out group with a saved booster (no fit). Returns (incidence dict, held-out rows)."""
    data, phys = build_data(tables["windows"], tables["events"], tables["movies"], hist, cfg, cfg_w, panel)
    te = data[data["group"] == group].copy().assign(censor_offset=np.inf)
    del data
    if te.empty:
        raise ValueError(f"no landmark of group {group}")
    tex, vcols = te, []
    if feats is not None:
        tex, vcols = join_store(te, rows, feats, VIDEO_PREFIX)
    cols = input_columns(method, vcols, phys)
    if booster.num_feature() != len(cols) + 1:
        raise ValueError(f"the booster reads {booster.num_feature()} inputs; {len(cols) + 1} expected")
    keys, Xt = base_matrix(tex, cols)
    out = predict_incidence(booster, Xt, cfg, cfg_w)
    out["_index"] = keys
    return out, te


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", "--fold", dest="group", required=True, choices=GROUPS,
                    help="the held-out group (--fold is the same option, as in the video commands)")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--inner-group", choices=GROUPS,
                    help="fit a cross-fitted inner head: on the seven groups other than --group and this group, "
                         "predicting this group, with the features of the model that held out --group")
    ap.add_argument("--panel", default="primary", choices=PANELS)
    ap.add_argument("--method", default="deadahead", choices=METHODS)
    ap.add_argument("--features", type=Path,
                    help="VideoMAE feature array (.npy); default: the fine-tuned model's features under runs/ "
                         "(required with --method videomae_adapted_survival: the array of deadahead video-features "
                         "--adapted)")
    ap.add_argument("--rows", type=Path, help="row index of the feature arrays (.parquet); default under runs/")
    ap.add_argument("--tables-dir", type=Path, help="folder with windows, events, movies and cells .parquet")
    ap.add_argument("--history", type=Path, help="precomputed history features (.parquet); default: built and cached")
    ap.add_argument("--untyped", default="censor_at_death", choices=("censor_at_death", "at_risk_to_cut"),
                    help="training labels of an untyped death: censored at the death (the paper), or followed up to "
                         "the truncation frame and censored there (written under its own file names)")
    ap.add_argument("--threads", type=int, default=8, help="LightGBM threads (the released heads used 8)")
    ap.add_argument("--booster", type=Path, help="predict with this saved booster instead of fitting one")
    ap.add_argument("--out-dir", type=Path, help="default: runs/<method>")
    a = ap.parse_args(argv)
    if a.method == "videomae_adapted_survival" and a.features is None:
        ap.error("--method videomae_adapted_survival needs --features: the features of the adapted encoder "
                 "(deadahead video-features --fold <group> --adapted writes runs/videomae/features/adapted/"
                 "<group>.npy); the default features are those of the fine-tuned model")
    if a.inner_group == a.group:
        ap.error("--inner-group: the held-out group is not one of its training groups")

    cfg, cfg_w = head_config(), windows_config()
    head.set_threads(a.threads)
    tables = load_tables(a.tables_dir, names=("windows", "events", "movies"))
    if a.history is not None:
        hist = pd.read_parquet(a.history)
    else:
        cells_path = (a.tables_dir / "cells.parquet") if a.tables_dir is not None else None
        hist = history_features(int(cfg_w["landmark"]["history_frames_W"]), cells_path)
    rows = feats = None
    if a.method != "handcrafted_survival":
        rows = pd.read_parquet(a.rows or default_rows())
        feats = load_features(a.features or default_features(a.group, a.seed), len(rows))

    out_dir = a.out_dir or runs_dir() / a.method
    tag = run_tag(a.untyped, a.inner_group)
    name = f"{a.method}__{a.group}__s{a.seed}__{a.panel}{tag}"
    if a.booster is not None:
        bp = a.booster
        out, te = predict_heldout(a.inner_group or a.group, load_booster(bp), tables, hist, cfg, cfg_w, panel=a.panel,
                                  method=a.method, rows=rows, feats=feats)
    else:
        out, booster, te, Xt = fit_head(a.group, a.seed, tables, hist, cfg, cfg_w, panel=a.panel, method=a.method,
                                        rows=rows, feats=feats, untyped=a.untyped, inner_group=a.inner_group)
        bp = save_booster(booster, out_dir / "boosters" / booster_name(a.method, a.group, a.seed, a.panel, tag))
        d = max_abs_diff(predict_incidence(load_booster(bp), Xt, cfg, cfg_w), out)
        if not d <= ROUNDTRIP_TOL:
            raise AssertionError(f"the saved booster does not reproduce the head's incidences ({d})")
    pay = prediction_payload(out, te)
    coh = coherence_violations(out)
    if coh:
        raise AssertionError(f"{coh} landmarks with an incidence that decreases with the horizon")
    pp = out_dir / "pred" / f"{name}.npz"
    pp.parent.mkdir(parents=True, exist_ok=True)
    tmp = pp.with_name(pp.name + ".partial.npz")
    np.savez_compressed(tmp, **pay)
    tmp.replace(pp)
    print(f"{name}: {len(out['_index'])} held-out landmarks -> {pp}; booster {bp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
