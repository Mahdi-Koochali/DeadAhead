"""DINOv2 + handcrafted through DeadAhead's survival head (the same-head comparison of the gap and fixed-horizon
results), one held-out group and seed at a time.

The head, its training labels and its settings are DeadAhead's (``deadahead.survival``); only the inputs differ: the
2,304 pooled DINOv2 values of the landmark (9 frames over the past 120 min, last, mean and change; the encoder adapted
without the held-out group) followed by the 112 handcrafted features, plus the bin index (2,417 inputs per
person-period row).

    deadahead baseline-dinov2-survival --group DU145_bp --seed 2601562
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from deadahead.baselines.heads import TEMPORAL, default_store, pool_embeddings
from deadahead.features.history import history_features
from deadahead.paths import runs_dir
from deadahead.survival import head
from deadahead.survival.boosters import (
    base_matrix,
    capture_booster,
    max_abs_diff,
    predict_incidence,
    save_booster,
)
from deadahead.survival.config import head_config, windows_config
from deadahead.survival.frames import (
    CUT_FRAME,
    GROUPS,
    KEY,
    N_HANDCRAFTED,
    PANELS,
    apply_view,
    assert_inside_cut,
    assert_same_rows,
    build_data,
    coherence_violations,
    landmark_table,
    load_tables,
    panel_view,
    panel_windows,
    prediction_payload,
    split_fold,
    store_positions,
)
from deadahead.survival.train import run_tag

METHOD = "dinov2_handcrafted_survival"
DINO_DIM = 768
BLOCK_I = 3 * DINO_DIM                                # 2,304 pooled DINOv2 values
WIDTH = BLOCK_I + N_HANDCRAFTED + 1                   # 2,417 with the bin index
ROUNDTRIP_TOL = 1e-9


def landmark_list(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame's landmarks in the frame's order with ``cell_num = int(track_id.split(":")[-1])``; raises unless the
    track to cell-number mapping is one-to-one within each movie."""
    lm = frame.drop_duplicates(subset=KEY)[KEY].reset_index(drop=True)
    try:
        lm["cell_num"] = lm["track_id"].astype(str).str.split(":").str[-1].astype(int)
    except (TypeError, ValueError) as e:
        raise ValueError(f"a track_id does not end in an integer cell number: {e}") from None
    pairs = lm[["movie_id", "track_id", "cell_num"]].drop_duplicates()
    if pairs.duplicated(["movie_id", "track_id"]).any() or pairs.duplicated(["movie_id", "cell_num"]).any():
        raise ValueError("the track_id <-> cell_num mapping is not one-to-one within a movie")
    return lm


def join_block_i(frame: pd.DataFrame, landmarks: pd.DataFrame, emb: np.ndarray,
                 cols: list[str]) -> pd.DataFrame:
    """The pooled DINOv2 block joined to ``frame`` by landmark key (``emb`` row i belongs to ``landmarks`` row i),
    keeping the row order."""
    if emb.shape != (len(landmarks), len(cols)):
        raise AssertionError(f"block I {emb.shape} against {len(landmarks)} landmarks x {len(cols)} names")
    pos = store_positions(frame, landmarks)
    block = pd.DataFrame(emb[pos], columns=cols, index=frame.index)
    out = pd.concat([frame, block], axis=1)
    assert_same_rows(frame, out, "the block I join")
    return out


def fit_unit(group: str, seed: int, tables: dict, hist: pd.DataFrame, cfg: dict, cfg_w: dict, store: Path, *,
             panel: str = "primary", untyped: str = "censor_at_death"):
    """Fit the survival head on [pooled DINOv2 | handcrafted] for one held-out group and seed.

    Returns (incidence dict of ``train_predict_survival``, booster, held-out rows).
    """
    data, phys = build_data(tables["windows"], tables["events"], tables["movies"], hist, cfg, cfg_w, panel)
    lm = landmark_table(panel_windows(tables["windows"], panel))
    tr0, te = split_fold(data, group)
    view = panel_view(lm, untyped, tables["movies"], CUT_FRAME)
    tr = apply_view(tr0, view)
    assert_inside_cut(tr, tables["movies"])
    te = te.assign(censor_offset=np.inf)                 # the held-out side is never censored
    lmi = landmark_list(data)
    emb, icols = pool_embeddings(lmi, TEMPORAL, Path(store))
    emb = np.asarray(emb)
    if emb.shape != (len(lmi), BLOCK_I) or emb.dtype != np.float32 or not np.isfinite(emb).all():
        raise ValueError(f"pooled DINOv2 block {emb.shape} {emb.dtype}; ({len(lmi)}, {BLOCK_I}) finite float32 expected")
    trx, tex = join_block_i(tr, lmi, emb, icols), join_block_i(te, lmi, emb, icols)
    del emb, data, tr0
    cols = list(icols) + list(phys)
    if len(cols) + 1 != WIDTH or len(set(cols)) != len(cols):
        raise AssertionError(f"{len(cols) + 1} inputs per person-period row, {WIDTH} expected")
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
    return out, boosters[0], te


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DINOv2 + handcrafted features through DeadAhead's survival head.")
    ap.add_argument("--group", "--fold", dest="group", required=True, choices=GROUPS)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--panel", default="primary", choices=PANELS)
    ap.add_argument("--store", type=Path, help="DINOv2 embedding store of the held-out group "
                                               "(default runs/dinov2/embeddings/<group>)")
    ap.add_argument("--tables-dir", type=Path, help="folder with windows, events, movies and cells .parquet")
    ap.add_argument("--history", type=Path, help="precomputed history features (.parquet); default: built and cached")
    ap.add_argument("--untyped", default="censor_at_death", choices=("censor_at_death", "at_risk_to_cut"))
    ap.add_argument("--threads", type=int, default=8, help="LightGBM threads (the released heads used 8)")
    ap.add_argument("--out-dir", type=Path, help=f"default: runs/{METHOD}")
    a = ap.parse_args(argv)
    cfg, cfg_w = head_config(), windows_config()
    head.set_threads(a.threads)
    tables = load_tables(a.tables_dir, names=("windows", "events", "movies"))
    if a.history is not None:
        hist = pd.read_parquet(a.history)
    else:
        cells_path = (a.tables_dir / "cells.parquet") if a.tables_dir is not None else None
        hist = history_features(int(cfg_w["landmark"]["history_frames_W"]), cells_path)
    store = a.store or default_store("dinov2", a.group)
    out, booster, te = fit_unit(a.group, a.seed, tables, hist, cfg, cfg_w, store, panel=a.panel, untyped=a.untyped)
    out_dir = a.out_dir or runs_dir() / METHOD
    name = f"{METHOD}__{a.group}__s{a.seed}__{a.panel}{run_tag(a.untyped)}"
    bp = save_booster(booster, out_dir / "boosters" / f"{name}.txt")
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
