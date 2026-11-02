"""Yeast predictions of a VideoMAE-large: a trained 2-class yeast model on its held-out position, or a released
3-class fine-tuned VideoMAE applied with no yeast training on every position.

    deadahead yeast-predict --heldout Pos3_3 \
        --weights runs/yeast/weights/videomae_public__Pos3_3__s2601562.pt
    deadahead yeast-predict --zero-shot \
        --weights weights/videomae_large_finetuned__PNT1A_bp__seed2601562.safetensors

The pass reads no label. Each position is read with the channel and the intensity map of the fold that holds it out.
Output columns: unit, position, t, p_none, p_death (2-class); the zero-shot pass adds p_apoptosis, p_lytic and
read_with_fold, with p_death = p_apoptosis + p_lytic.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from . import constants as C
from . import data as D


def zero_shot(model, store: D.VidStore, keys, folds: dict, src_m: dict, mean, std, dev, *,
              positions=C.POSITIONS):
    """A 3-class model on every landmark of ``positions``, each read with the fold that holds it out:
    P(death within 30 min) = P(apoptosis) + P(lytic)."""
    import pandas as pd

    from . import model as M
    parts = []
    for p in positions:                                  # each position under the fold that holds it out
        kp = keys[keys["position"] == p].reset_index(drop=True)
        probs, _ = M.predict(model, store, kp, folds[p], src_m, mean, std, dev)
        parts.append(kp.assign(p_none=probs[:, 0], p_apoptosis=probs[:, 1], p_lytic=probs[:, 2],
                               p_death=probs[:, 1] + probs[:, 2], read_with_fold=p))
    return pd.concat(parts, ignore_index=True)


def held_out(model, store: D.VidStore, keys, folds: dict, src_m: dict, mean, std, dev, heldout: str):
    """A trained 2-class yeast model on its held-out position: p_none and p_death per landmark."""
    from . import model as M
    kp = keys[keys["position"] == heldout].reset_index(drop=True)
    probs, _ = M.predict(model, store, kp, folds[heldout], src_m, mean, std, dev)
    return kp.assign(p_none=probs[:, 0], p_death=probs[:, 1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Yeast predictions of a VideoMAE-large (no label is read).")
    ap.add_argument("--weights", type=Path, required=True,
                    help="a trained yeast model (.pt), or with --zero-shot a released .safetensors file")
    ap.add_argument("--heldout", choices=list(C.POSITIONS), default=None,
                    help="held-out position of a trained yeast model")
    ap.add_argument("--zero-shot", action="store_true", help="a released 3-class model on every position")
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="per-fold inputs (default: configs/yeast.yaml)")
    ap.add_argument("--rows-dir", type=Path, default=None)
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="output parquet (default: runs/yeast/pred/<name>.parquet)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    if not a.zero_shot and a.heldout is None:
        ap.error("a trained yeast model needs --heldout")
    if not a.weights.is_file():
        ap.error(f"--weights: no file {a.weights}")

    from .finetune import (default_config, default_out_dir, default_rows_dir, default_store, read_inputs,
                           require_torch)
    require_torch(ap)

    import pandas as pd
    import torch

    from deadahead.video.clips import input_missing
    from deadahead.video.model import image_stats, load_released_classifier, pretrained_snapshot

    from . import model as M

    keys_file = (a.rows_dir or default_rows_dir()) / "all_keys.parquet"
    if input_missing(keys_file, "yeast-data"):
        return 3
    folds, src_m = read_inputs(a.config or default_config())
    keys = pd.read_parquet(keys_file)
    store = D.VidStore(a.store or default_store())
    snap = a.snapshot or pretrained_snapshot()
    mean, std = image_stats(snap)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    stem = a.weights.name.split(".")[0]
    if a.zero_shot:
        model = load_released_classifier(a.weights, snap, dev)
        out = zero_shot(model, store, keys, folds, src_m, mean, std, dev)
        cols = ["p_none", "p_apoptosis", "p_lytic"]
        name = f"zero_shot__{stem}"
    else:
        model = M.load_yeast_classifier(a.weights, snap, dev)
        out = held_out(model, store, keys, folds, src_m, mean, std, dev, a.heldout)
        cols = ["p_none", "p_death"]
        name = stem
    if not np.isfinite(out[cols].to_numpy(float)).all():
        print("some probabilities are not finite")
        return 5
    dst = a.out or default_out_dir() / "pred" / f"{name}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp{os.getpid()}")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, dst)
    print(f"{len(out)} rows -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
