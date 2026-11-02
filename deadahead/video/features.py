"""Per-row features of VideoMAE (1,024 values) for every landmark row of the primary panel.

Fine-tuned model: the classifier's input, ``fc_norm`` of the token mean. Adapted encoder (``--adapted``),
before any fine-tuning: the mean over tokens of its last hidden state, after its own final LayerNorm.

One checkpoint embeds all rows, in the order of ``clips.read_landmarks`` (movie, track, frame). The
pass is the prediction path (no augmentation). Output under ``runs/videomae/features/``:
``<group>__s<seed>.npy`` (N, 1024) float32, ``<group>__s<seed>.logits.npy`` (N, 3) float32,
``adapted/<group>.npy`` (N, 1024) float32 and ``rows.parquet`` (movie_id, track_id, t_obs_frame, group).

    deadahead video-features --fold DU145_bp --seed 2601562 \
        --weights videomae_large_finetuned__DU145_bp__seed2601562.safetensors
    deadahead video-features --fold DU145_bp --adapted
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from deadahead import paths

from .adapt import default_out_dir as default_adapt_dir
from .clips import build_group_index, default_store, default_windows, input_missing, read_landmarks
from .finetune import checkpoint_path
from .finetune import default_out_dir as default_finetune_dir
from .model import InputError, check_weights_name

PASS_BATCH = 64
N_LOGITS = 3
ATTENTION = "sdpa"
ROW_COLUMNS = ("movie_id", "track_id", "t_obs_frame", "group")


def default_out_dir() -> Path:
    return paths.runs_dir() / "videomae" / "features"


def rows_with_clips(windows: Path, store: Path):
    """Every landmark row with its ``frame_rows``, in ``read_landmarks`` order."""
    import pandas as pd
    rows = read_landmarks(windows)[list(ROW_COLUMNS)]
    keys = ["movie_id", "track_id", "t_obs_frame"]
    parts = []
    for g in sorted(rows["group"].unique()):
        fc = build_group_index(rows, store, g)
        if fc.dropped_missing_frames:
            raise ValueError(f"{fc.dropped_missing_frames} rows of {g} lack a frame in the store")
        parts.append(fc.clips[keys + ["frame_rows"]])
    clips = pd.concat(parts, ignore_index=True)
    out = rows.merge(clips, on=keys, how="left", validate="one_to_one")
    if out["frame_rows"].isna().any():
        raise ValueError("a row of the index has no clip")
    return out


def model_inputs(gather, rows, pick, mean, std, dev):
    """Model input for rows ``pick``: gather, phase map, resize and normalise; no augmentation."""
    import torch
    from deadahead.video import training as T
    from deadahead.video.model import FRAMES
    movies, frs = rows["movie_id"].to_numpy(), rows["frame_rows"].to_numpy()
    raw = torch.from_numpy(np.ascontiguousarray(gather.batch(movies[pick], [frs[i] for i in pick])))
    x = T.phase_to_unit(raw.to(dev, non_blocking=True))
    if x.shape[1] != FRAMES:
        raise ValueError(f"a clip has {x.shape[1]} frames, not {FRAMES}")
    return T.to_model_input(x, mean, std)


def bare_encoder(snap: Path):
    """A bare ``VideoMAEModel`` from the checkpoint's config; without mean pooling it ends in its final LayerNorm."""
    from transformers import VideoMAEConfig, VideoMAEModel
    cfg = VideoMAEConfig.from_pretrained(str(snap), attn_implementation=ATTENTION)
    if cfg.use_mean_pooling:
        raise InputError("the checkpoint's config has use_mean_pooling true; the adapted encoder needs false")
    return VideoMAEModel(cfg)


def load_adapted(adapted: Path, snap: Path, dev):
    """The adapted encoder of an ``adapt__<group>.pt`` in a bare ``VideoMAEModel``, loaded strictly."""
    import torch
    sd = torch.load(adapted, map_location="cpu", weights_only=False)["state_dict"]
    model = bare_encoder(snap)
    model.load_state_dict(sd, strict=True)
    return model.to(dev).eval()


def adapted_features(encoder, rows, store: Path, mean, std, dev, *, batch: int = PASS_BATCH,
                     deadline: float | None = None) -> np.ndarray:
    """The mean over all tokens of the last hidden state after the encoder's own final LayerNorm:
    (N, 1024) float32, in row order."""
    import torch
    from deadahead.video.clips import ClipGatherer
    g = ClipGatherer(store)
    out = np.empty((len(rows), encoder.config.hidden_size), dtype=np.float32)   # 1,024 for the pinned model
    try:
        with torch.no_grad():
            for s in range(0, len(rows), batch):
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(f"the pass reached its time limit after {s} of {len(rows)} rows")
                pick = np.arange(s, min(s + batch, len(rows)))
                inp = model_inputs(g, rows, pick, mean, std, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    h = encoder(pixel_values=inp).last_hidden_state
                out[pick] = h.float().mean(dim=1).cpu().numpy()
    finally:
        g.close()
    return out


def finetuned_features(model, rows, store: Path, mean, std, dev, *, batch: int = PASS_BATCH,
                       deadline: float | None = None):
    """The classifier's input, ``fc_norm`` of the mean over tokens, captured by a forward hook on
    ``classifier`` in the pass that gives the logits, and the classifier re-applied to it.

    Returns (features (N, 1024), logits (N, 3), re-applied logits (N, 3)), all float32.
    """
    import torch
    from deadahead.video.clips import ClipGatherer
    g = ClipGatherer(store)
    n = len(rows)
    feats = np.empty((n, model.config.hidden_size), dtype=np.float32)           # 1,024 for the pinned model
    logits = np.empty((n, N_LOGITS), dtype=np.float32)
    again = np.empty((n, N_LOGITS), dtype=np.float32)
    seen = {}
    hook = model.classifier.register_forward_hook(lambda mod, args, out: seen.__setitem__("x", args[0]))
    try:
        with torch.no_grad():
            for s in range(0, n, batch):
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(f"the pass reached its time limit after {s} of {n} rows")
                pick = np.arange(s, min(s + batch, n))
                inp = model_inputs(g, rows, pick, mean, std, dev)
                seen.clear()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    lg = model(pixel_values=inp).logits
                    x = seen["x"]
                    lg2 = model.classifier(x)
                feats[pick] = x.float().cpu().numpy()
                logits[pick] = lg.float().cpu().numpy()
                again[pick] = lg2.float().cpu().numpy()
    finally:
        hook.remove()
        g.close()
    return feats, logits, again


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-row 1,024-d features of a fine-tuned or adapted VideoMAE.")
    ap.add_argument("--fold", required=True, help="held-out group of the checkpoint")
    ap.add_argument("--seed", type=int, default=None, help="seed of the fine-tuned model")
    ap.add_argument("--weights", type=Path, default=None, help="a released .safetensors file")
    ap.add_argument("--backbone-lr", type=float, default=None, help="setting of a .pt final run")
    ap.add_argument("--layer-decay", type=float, default=None)
    ap.add_argument("--adapted", action="store_true",
                    help="features of the group's adapted encoder, before fine-tuning, in <out-dir>/adapted/")
    ap.add_argument("--adapt-dir", type=Path, default=None, help="folder of adapt__<group>.pt")
    ap.add_argument("--batch", type=int, default=PASS_BATCH)
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--runs-dir", type=Path, default=None)
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--windows", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    if a.adapted and any(v is not None for v in (a.seed, a.weights, a.backbone_lr, a.layer_decay)):
        ap.error("--adapted takes no --seed, --weights, --backbone-lr or --layer-decay")
    if not a.adapted and a.seed is None:
        ap.error("give --seed, or --adapted")
    if not a.adapted and a.weights is None and (a.backbone_lr is None or a.layer_decay is None):
        ap.error("give --weights, or --backbone-lr and --layer-decay of the final run")
    if a.weights is not None:
        check_weights_name(a.weights, a.fold, a.seed)
    adapted = (a.adapt_dir or default_adapt_dir()) / f"adapt__{a.fold}.pt"
    windows = a.windows or default_windows()
    if input_missing(windows, "data-windows") or (a.adapted and input_missing(adapted, "video-adapt")):
        return 3

    from .model import image_stats, load_classifier, load_released_classifier, pretrained_snapshot, resolve_device

    store = a.store or default_store()
    out_dir = a.out_dir or default_out_dir()
    rows = rows_with_clips(windows, store)
    snap = a.snapshot or pretrained_snapshot()
    mean, std = image_stats(snap)
    dev = resolve_device(a.device)
    if a.adapted:
        feats = adapted_features(load_adapted(adapted, snap, dev), rows, store, mean, std, dev, batch=a.batch)
        if not np.isfinite(feats).all():
            raise FloatingPointError("non-finite features")
        out = out_dir / "adapted" / f"{a.fold}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, feats)
        rows[list(ROW_COLUMNS)].to_parquet(out_dir / "rows.parquet", index=False)
        print(f"{len(rows)} rows x {feats.shape[1]} -> {out}")
        return 0
    if a.weights is not None:
        model = load_released_classifier(a.weights, snap, dev)
    else:
        ckpt = checkpoint_path(a.runs_dir or default_finetune_dir(), a.fold, a.backbone_lr, a.layer_decay,
                               a.seed, final=True)
        model = load_classifier(ckpt, snap, dev)
    feats, logits, again = finetuned_features(model, rows, store, mean, std, dev, batch=a.batch)
    if not (np.isfinite(feats).all() and np.isfinite(logits).all()):
        raise FloatingPointError("non-finite features or logits")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{a.fold}__s{a.seed}.npy", feats)
    np.save(out_dir / f"{a.fold}__s{a.seed}.logits.npy", logits)
    rows[list(ROW_COLUMNS)].to_parquet(out_dir / "rows.parquet", index=False)
    print(f"{len(rows)} rows x {feats.shape[1]} -> {out_dir / f'{a.fold}__s{a.seed}.npy'}; "
          f"max |logits - classifier(features)| {float(np.abs(logits - again).max()):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
