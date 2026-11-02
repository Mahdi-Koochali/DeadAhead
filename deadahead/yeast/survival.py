"""The survival layer of one fine-tuned yeast run: the one-cause survival head (``rules.survival_cif``: bins of 6 frames,
8 bins, 200 rounds, weight 1, the LightGBM seed is the run's seed) fitted on the training positions' landmarks embedded
by that run's own model, scored on the predicted position (the held-out position; the threshold position for a
threshold run). Rows:
  S    the 1,024-d embeddings alone;
  SL   the embeddings plus the label features;
  NB   the embeddings with two causes, the next bud and death (the next-bud split).
The fit reads only the fold's survival rows (event and last-observed offsets of the training positions), never a
held-out label. Writes <runs>/yeast/surv/<run>__{S,SL,NB}.parquet (cumulative incidence at 30 / 60 / 120 / 240 min).

    deadahead yeast-survival --run videomae_finetuned__Pos3_3__s2601562
    deadahead yeast-survival --run thr__videomae_public__Pos3_3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from deadahead import paths
from deadahead.video.clips import input_missing

from . import constants as C

ROWS = ("S", "SL")
NB_ROW = "NB"                              # next bud against death, two causes


def fit_unit(emb, emb_keys, surv_rows, feats, predict_keys, seed: int) -> dict:
    """{row: frame of predict_keys with cif_<h>} for the S, SL and NB rows. Pure given its inputs."""
    from . import data as D
    from . import rules as B
    pos_of = {(u, t): i for i, (u, t) in enumerate(zip(emb_keys["unit"], emb_keys["t"]))}
    tr_idx = np.array([pos_of[(u, t)] for u, t in zip(surv_rows["unit"], surv_rows["t"])])
    te_idx = np.array([pos_of[(u, t)] for u, t in zip(predict_keys["unit"], predict_keys["t"])])
    if set(surv_rows["position"]) & set(predict_keys["position"]):
        raise ValueError("a predicted position is among the survival layer's fit rows")
    ev_bin, cens_bin = D.survival_bins(surv_rows)
    cause = np.isfinite(ev_bin).astype(int)
    f = feats.set_index(["unit", "t"])[list(B.LABEL_FEATURES)]
    lab_tr = f.loc[list(zip(surv_rows["unit"], surv_rows["t"]))].to_numpy(float)
    lab_te = f.loc[list(zip(predict_keys["unit"], predict_keys["t"]))].to_numpy(float)
    out = {}
    for row in ROWS:
        Xtr, Xte = emb[tr_idx], emb[te_idx]
        if row == "SL":
            Xtr, Xte = np.hstack([Xtr, lab_tr]), np.hstack([Xte, lab_te])
        cif = B.survival_cif(Xtr, ev_bin, cause, cens_bin, Xte, seed=seed, n_causes=1)
        fr = predict_keys[["unit", "position", "t"]].reset_index(drop=True).copy()
        for h, k in B.BIN_OF_H.items():
            fr[f"cif_{h}"] = cif[k][:, 0] if cif is not None else np.nan
        out[row] = fr
    ob, oc, ocens = D.bud_death_bins(surv_rows)
    cif = B.survival_cif(emb[tr_idx], ob, oc, ocens, emb[te_idx], seed=seed, n_causes=2)
    fr = predict_keys[["unit", "position", "t"]].reset_index(drop=True).copy()
    for h, k in B.BIN_OF_H.items():
        fr[f"cif_bud_{h}"] = cif[k][:, 0] if cif is not None else np.nan
        fr[f"cif_death_{h}"] = cif[k][:, 1] if cif is not None else np.nan
    out[NB_ROW] = fr
    return out


def parse_run(name: str) -> dict:
    """``<start>__<position>__s<seed>`` (a final run) or ``thr__<start>__<position>`` (a threshold run)."""
    parts = name.split("__")
    if len(parts) == 3 and parts[0] == "thr":
        out = {"kind": "thr", "start": parts[1], "heldout": parts[2], "seed": C.SEED}
    elif len(parts) == 3 and parts[2].startswith("s"):
        out = {"kind": "final", "start": parts[0], "heldout": parts[1], "seed": int(parts[2][1:])}
    else:
        raise ValueError(f"not a final or threshold run name: {name!r}")
    if out["start"] not in C.STARTS or out["heldout"] not in C.POSITIONS:
        raise ValueError(f"unknown start or position in {name!r}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a final run (<start>__<position>__s<seed>) or thr__<start>__<position>")
    ap.add_argument("--rows-dir", type=Path, default=None, help="row files (default: <DEADAHEAD_DATA>/yeast/rows)")
    ap.add_argument("--runs-dir", type=Path, default=None, help="yeast runs (default: <DEADAHEAD_DATA>/runs/yeast)")
    a = ap.parse_args(argv)
    try:
        u = parse_run(a.run)
    except ValueError as e:
        ap.error(str(e))
    rows_dir = a.rows_dir or (paths.data_root() / "yeast" / "rows")
    runs = a.runs_dir or (paths.runs_dir() / "yeast")
    kind = "surv" if u["kind"] == "final" else "survthr"
    surv_file, ef = rows_dir / f"{kind}__{u['heldout']}.parquet", runs / "embed" / f"{a.run}.npy"
    if input_missing(surv_file, "yeast-data") or input_missing(ef, "yeast-finetune"):
        return 3
    surv_rows = pd.read_parquet(surv_file)
    want = C.training_positions(u["heldout"]) if u["kind"] == "final" else C.threshold_run_positions(u["heldout"])
    if sorted(surv_rows["position"].unique()) != sorted(want):
        raise ValueError(f"{kind}__{u['heldout']} holds {sorted(surv_rows['position'].unique())}, not {sorted(want)}")
    emb = np.load(ef)
    emb_keys = pd.read_parquet(ef.with_suffix(".keys.parquet"))
    pred_pos = u["heldout"] if u["kind"] == "final" else C.threshold_position(u["heldout"])
    predict_keys = emb_keys[emb_keys["position"] == pred_pos].reset_index(drop=True)
    out = fit_unit(emb, emb_keys, surv_rows, pd.read_parquet(rows_dir / "label_features.parquet"), predict_keys,
                   u["seed"])
    for row, fr in out.items():
        dst = runs / "surv" / f"{a.run}__{row}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + f".tmp{os.getpid()}")
        fr.to_parquet(tmp, index=False)
        os.replace(tmp, dst)
    print(f"{a.run}: {', '.join(out)} on {pred_pos}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
