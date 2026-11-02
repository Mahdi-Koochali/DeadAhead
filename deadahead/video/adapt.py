"""Masked-autoencoder adaptation of VideoMAE-large on one fold's unlabelled training clips.

The checkpoint continues its own objective (tube masking at 90 %, loss on normalised pixels of the
masked patches) on clips of the fold's eight training groups; the held-out group contributes nothing
and no label is read. The decoder is trained and then discarded: only the encoder is saved, as
``adapt__<group>.pt``. Re-running the same command resumes from the last saved state; a command with
other ``--steps`` or ``--seed`` refuses the files of the earlier run until they are deleted.

    deadahead video-adapt --fold DU145_bp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deadahead import paths

from .clips import (ClipBuildError, ClipGatherer, assert_no_label_columns, default_clips_dir, default_store,
                    input_missing)
from .model import MASK_RATIO

#: Recipe of the adaptation stage.
STEPS = 6000
LR = 7.5e-5            # the checkpoint's pretraining rate scaled to batch 128
WEIGHT_DECAY = 0.05
WARMUP_FRACTION = 0.05
SEED = 2601562
MICRO_BATCH = 16       # 16 x 8 = 128; 32 x 4 fits on GPUs with more than 80 GB


def default_out_dir() -> Path:
    return paths.runs_dir() / "videomae" / "adapt"


def held_out_rows(idx, fold: str) -> tuple[int, int]:
    """Rows of a training index that belong to the held-out group, by group and by movie-id prefix."""
    held_rows = int((idx["group"] == fold).sum())
    held_movies = int(idx["movie_id"].astype(str).str.startswith(f"{fold}_").sum())
    return held_rows, held_movies


def train(model, idx, store, mean, std, dev, *, steps: int, split, seed: int, state: Path, run: dict,
          ckpt_every: int = 500, log=print):
    """Run the adaptation loop; returns the per-step losses, or None if interrupted (state saved).

    ``model`` is a ``VideoMAEForPreTraining`` already on ``dev``; ``idx`` is a fold's clip index; ``run``
    identifies the run in its state file.
    """
    import torch

    from . import training as T
    from .training import tube_mask

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = T.warmup_cosine(opt, steps, WARMUP_FRACTION)
    start, rng = 0, np.random.default_rng(seed)
    g_aug = torch.Generator(device=dev).manual_seed(seed + T.AUG_STREAM_OFFSET)
    K = T.blur_kernel(dev)
    losses: list[float] = []
    if state.exists():
        start, rng, losses = T.load_state(state, model, opt, sched, g_aug, dev, run=run)
        log(f"  resumed from step {start} of {steps}")

    gather = ClipGatherer(store)
    movies = idx["movie_id"].to_numpy()
    rows = idx["frame_rows"].to_numpy()
    stop = T.Interrupt()
    log_every = max(1, min(50, steps // 4))
    t0 = T.now()
    for step in range(start, steps):
        opt.zero_grad(set_to_none=True)
        step_loss, step_finite = 0.0, True
        for _ in range(split.grad_accum):
            pick = rng.integers(0, len(idx), size=split.micro_batch)
            x = T.to_model_input(T.clip_batch(gather, movies, rows, pick, dev, g_aug, K), mean, std)
            m = torch.from_numpy(tube_mask(len(pick), rng, ratio=MASK_RATIO)).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = model(pixel_values=x, bool_masked_pos=m).loss / split.grad_accum
            finite = bool(torch.isfinite(loss))
            step_finite = step_finite and finite
            step_loss += float(loss.detach())
            loss.backward()                  # no loss scaler: bf16 needs none, and a non-finite step stays visible
        opt.step(); sched.step()
        losses.append(step_loss)
        if not step_finite:
            log(f"  step {step + 1}: loss is not finite")
        if (step + 1) % log_every == 0:
            log(f"  step {step + 1:>5}/{steps}  loss {step_loss:.4f}  {(step + 1 - start) / (T.now() - t0):.2f} steps/s")
        if ckpt_every and (step + 1) % ckpt_every == 0 and step + 1 < steps:
            T.save_state(state, model, opt, sched, step + 1, rng, g_aug, losses, run=run)
        if stop.hit:
            T.save_state(state, model, opt, sched, step + 1, rng, g_aug, losses, run=run)
            log(f"  interrupted at step {step + 1}; state saved, re-run to resume")
            gather.close()
            return None
    if ckpt_every and steps > start:
        T.save_state(state, model, opt, sched, steps, rng, g_aug, losses, run=run)
    gather.close()
    return losses


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Masked-autoencoder adaptation of VideoMAE-large on one fold.")
    ap.add_argument("--fold", required=True, help="held-out group; the other eight groups are used")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--micro-batch", type=int, default=MICRO_BATCH, help="rows per forward pass; must divide 128")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ckpt-every", type=int, default=500, help="save resumable state every N steps")
    ap.add_argument("--snapshot", type=Path, default=None,
                    help="local directory of the checkpoint (default: download the pinned revision)")
    ap.add_argument("--clips-dir", type=Path, default=None)
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    a = ap.parse_args(argv)

    import pandas as pd
    import torch

    from . import training as T
    from .model import build_pretraining_model, checkpoint_tensors, image_stats, pretrained_snapshot, resolve_device

    clips_dir = a.clips_dir or default_clips_dir()
    store = a.store or default_store()
    out_dir = a.out_dir or default_out_dir()
    split = T.split_batch(a.micro_batch)
    ckpt = out_dir / f"adapt__{a.fold}.pt"
    run = {"steps": a.steps, "seed": a.seed, "effective": split.effective}
    if ckpt.is_file():
        return T.finished_run(ckpt, run)
    clip_index = clips_dir / f"clips__{a.fold}.parquet"
    if input_missing(clip_index, "video-clips"):
        return 3

    idx = pd.read_parquet(clip_index)
    assert_no_label_columns(idx.columns)
    if any(held_out_rows(idx, a.fold)):
        raise ClipBuildError(f"the clip index of fold {a.fold} holds rows of the held-out group")
    print(f"adapt fold {a.fold}: {len(idx)} clips from {idx['movie_id'].nunique()} movies, "
          f"{idx['group'].nunique()} training groups; {a.steps} steps of "
          f"{split.micro_batch} x {split.grad_accum}", flush=True)

    snap = a.snapshot or pretrained_snapshot()
    mean, std = image_stats(snap)
    ck = checkpoint_tensors(snap)
    dev = resolve_device(a.device)
    torch.manual_seed(a.seed)                  # before the model is built
    model = build_pretraining_model(snap, ck)  # attention biases restored on the CPU, then moved
    model = model.to(dev)

    out_dir.mkdir(parents=True, exist_ok=True)
    state = out_dir / f"adapt__{a.fold}.state.pt"
    losses = train(model, idx, store, mean, std, dev, steps=a.steps, split=split, seed=a.seed,
                   state=state, run=run, ckpt_every=a.ckpt_every, log=lambda s: print(s, flush=True))
    if losses is None:
        return 10
    # the record comes first, so a checkpoint never exists without it
    ckpt.with_suffix(".json").write_text(json.dumps(
        {"fold": a.fold, **run, "micro_batch": split.micro_batch,
         "grad_accum": split.grad_accum, "lr": LR, "weight_decay": WEIGHT_DECAY,
         "all_steps_finite": bool(np.isfinite(losses).all()),
         "mean_loss_first_100": float(np.mean(losses[:100])) if losses else None,
         "mean_loss_last_100": float(np.mean(losses[-100:])) if losses else None}, indent=2) + "\n")
    T.atomic_save({"state_dict": {k: v for k, v in model.videomae.state_dict().items()}},
                  ckpt)                           # encoder only: the decoder is discarded
    print(f"wrote {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
