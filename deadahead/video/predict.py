"""Held-out predictions of a fine-tuned VideoMAE: three class probabilities per 30-minute row.

The input path is the training path without augmentation: gather the 16 frames, apply the phase
map, resize to 224 and normalise. bf16 autocast on CUDA as in training; softmax in float32. The rows
scored must be exactly the held-out group's labelled rows. Output:
``runs/videomae/pred/<group>__s<seed>.parquet`` with columns ``PRED_COLUMNS``.

    deadahead video-predict --fold DU145_bp --seed 2601562 --backbone-lr 2e-4 --layer-decay 0.9
    deadahead video-predict --fold DU145_bp --seed 2601562 \
        --weights videomae_large_finetuned__DU145_bp__seed2601562.safetensors
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from deadahead import paths

from .clips import build_group_index, default_store, default_windows, input_missing, read_landmarks
from .finetune import CLASSES, checkpoint_path, label_table
from .finetune import default_out_dir as default_finetune_dir
from .model import check_weights_name

#: Forward batch.
PRED_BATCH = 64
PRED_COLUMNS = ("movie_id", "track_id", "t_obs_frame", "label", "p_none", "p_apoptosis", "p_lytic")


def default_out_dir() -> Path:
    return paths.runs_dir() / "videomae" / "pred"


def _keys(df) -> list[tuple]:
    return list(zip(df["movie_id"].astype(str), df["track_id"].astype(str), df["t_obs_frame"].astype(int)))


def held_out_row_problems(scored, expected, group: str) -> list[str]:
    """Differences between the rows to be scored and the held-out group's labelled rows; empty if none.

    A landmark with a frame missing from the store would otherwise be dropped silently by the join.
    """
    out = []
    s_keys, e_keys = _keys(scored), _keys(expected)
    s, e = set(s_keys), set(e_keys)
    if len(s) != len(s_keys):
        out.append(f"{len(s_keys) - len(s)} scored rows are duplicated")
    if len(e) != len(e_keys):
        out.append(f"{len(e_keys) - len(e)} of label_table's rows for {group} are duplicated")
    missing, extra = sorted(e - s), sorted(s - e)
    if missing:
        out.append(f"{len(missing)} of the {len(e)} labelled rows of {group} would not be scored "
                   f"(e.g. {missing[:3]})")
    if extra:
        out.append(f"{len(extra)} scored rows are not among label_table's rows for {group} (e.g. {extra[:3]})")
    return out


def predict_probs(model, clips, store: Path, mean, std, dev, *, batch: int = 64,
                  check_finite: bool = True, deadline: float | None = None) -> np.ndarray:
    """``(N, 3)`` softmax probabilities for the rows of ``clips``, in row order.

    ``deadline`` (a ``time.monotonic()`` value) raises TimeoutError between batches once passed.
    """
    import torch
    from deadahead.video import training as T
    from deadahead.video.clips import ClipGatherer
    from deadahead.video.model import FRAMES

    gather = ClipGatherer(store)
    movies = clips["movie_id"].to_numpy()
    rows = clips["frame_rows"].to_numpy()
    out = np.empty((len(clips), 3), dtype=np.float64)
    try:
        with torch.no_grad():
            for s in range(0, len(clips), batch):
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(f"the pass reached its time limit after {s} of {len(clips)} rows")
                pick = np.arange(s, min(s + batch, len(clips)))
                raw = torch.from_numpy(np.ascontiguousarray(
                    gather.batch(movies[pick], [rows[i] for i in pick])))
                x = T.phase_to_unit(raw.to(dev, non_blocking=True))
                if x.shape[1] != FRAMES:
                    raise ValueError(f"a clip has {x.shape[1]} frames, not {FRAMES}")
                # no augmentation at test time
                inp = T.to_model_input(x, mean, std)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    logits = model(pixel_values=inp).logits
                out[pick] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    finally:
        gather.close()
    # check_finite=False returns non-finite probabilities of a diverged model instead of raising.
    if check_finite and not np.isfinite(out).all():
        raise FloatingPointError("the forward pass produced non-finite probabilities")
    return out


def prediction_frame(rows, p: np.ndarray, classes):
    """Per-row prediction table with columns ``PRED_COLUMNS``, in row order."""
    import pandas as pd
    p = np.asarray(p, dtype=float)
    if p.shape != (len(rows), 3):
        raise ValueError(f"expected p of shape ({len(rows)}, 3), got {p.shape}")
    out = pd.DataFrame({
        "movie_id": rows["movie_id"].to_numpy(), "track_id": rows["track_id"].to_numpy(),
        "t_obs_frame": rows["t_obs_frame"].to_numpy(),
        "label": [classes[int(v)] for v in rows["y"]],
        "p_none": p[:, 0], "p_apoptosis": p[:, 1], "p_lytic": p[:, 2]})
    return out[list(PRED_COLUMNS)]


def held_out_rows(windows: Path, store: Path, fold: str):
    """The held-out group's labelled rows with their clips, in clip-index order."""
    labels = label_table(windows)
    held = build_group_index(read_landmarks(windows), store, fold)
    rows = held.clips.merge(labels, on=["movie_id", "track_id", "t_obs_frame", "group"],
                            how="inner").reset_index(drop=True)
    expected = labels[labels["group"] == fold]
    problems = held_out_row_problems(rows, expected, fold)
    if problems:
        raise ValueError("; ".join(problems))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Held-out predictions of a fine-tuned VideoMAE.")
    ap.add_argument("--fold", required=True, help="held-out group")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--backbone-lr", type=float, default=None, help="setting of the final run (with --layer-decay)")
    ap.add_argument("--layer-decay", type=float, default=None)
    ap.add_argument("--weights", type=Path, default=None, help="a released .safetensors file instead of a .pt")
    ap.add_argument("--batch", type=int, default=PRED_BATCH)
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--runs-dir", type=Path, default=None, help="fine-tuning checkpoints")
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--windows", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    if a.weights is None and (a.backbone_lr is None or a.layer_decay is None):
        ap.error("give --weights, or --backbone-lr and --layer-decay of the final run")
    if a.weights is not None:
        check_weights_name(a.weights, a.fold, a.seed)

    from .model import image_stats, load_classifier, load_released_classifier, pretrained_snapshot, resolve_device

    store = a.store or default_store()
    windows = a.windows or default_windows()
    out_dir = a.out_dir or default_out_dir()
    if input_missing(windows, "data-windows"):
        return 3
    rows = held_out_rows(windows, store, a.fold)
    snap = a.snapshot or pretrained_snapshot()
    mean, std = image_stats(snap)
    dev = resolve_device(a.device)
    if a.weights is not None:
        model = load_released_classifier(a.weights, snap, dev)
    else:
        ckpt = checkpoint_path(a.runs_dir or default_finetune_dir(), a.fold, a.backbone_lr, a.layer_decay,
                               a.seed, final=True)
        model = load_classifier(ckpt, snap, dev)
    p = predict_probs(model, rows, store, mean, std, dev, batch=a.batch)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a.fold}__s{a.seed}.parquet"
    prediction_frame(rows, p, CLASSES).to_parquet(out, index=False)
    print(f"{len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
