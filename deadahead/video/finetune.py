"""Fine-tune the whole VideoMAE-large end to end on 30-minute death labels, starting from the fold's
adapted encoder.

Three classes (no death, apoptosis, lytic death) from the forecast rows of the primary panel;
censored rows are dropped, never counted as negatives. Layer-wise learning-rate decay is applied
through optimiser parameter groups. Each fold first trains the four settings (backbone rate x layer
decay) on seven training groups, holding out the training group with the most lytic events for
selection (``deadahead.video.selection``); the chosen setting is then retrained on all eight training
groups (``--final``) for three seeds. Re-running a command resumes from the last saved state; a command
with other ``--steps``, or a changed adapted encoder, refuses the files of the earlier run until they are
deleted.

    deadahead video-finetune --fold DU145_bp --backbone-lr 2e-4 --layer-decay 0.9
    deadahead video-finetune --fold DU145_bp --backbone-lr 2e-4 --layer-decay 0.9 --final --seed 2601563
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deadahead import paths

from .adapt import default_out_dir as default_adapt_dir
from .clips import ClipBuildError, ClipGatherer, default_clips_dir, default_store, default_windows, input_missing
from .model import CLASSES

#: Recipe of the fine-tuning stage.
HORIZON = 30
PANEL = "primary_439"
EXCLUDED_MOVIES = ("DU145_st_2",)          # left out of fine-tuning and scoring; the adaptation still reads it
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.05
WARMUP_FRACTION = 0.05
GRAD_CLIP = 1.0
SEED = 2601562
FINAL_SEEDS = (2601562, 2601563, 2601564)
STEPS = 6000
MICRO_BATCH = 16       # 16 x 8 = 128; 32 x 4 fits on GPUs with more than 80 GB
#: The four settings in preference order; an exact tie in selection goes to the earlier one.
GRID = [(5e-5, 0.75), (5e-5, 0.9), (2e-4, 0.75), (2e-4, 0.9)]


def default_out_dir() -> Path:
    return paths.runs_dir() / "videomae" / "finetune"


def label_table(windows_parquet: Path):
    """30-minute labels of the primary panel, three classes, censored rows absent.

    ``CENSORED_IN_HORIZON`` rows are dropped; an untyped death counts as no event.
    """
    import pandas as pd
    w = pd.read_parquet(windows_parquet, columns=["movie_id", "track_id", "t_obs_frame", "panel",
                                                  "group", "horizon_min", "label", "window_weight"])
    w = w[(w["panel"] == PANEL) & (w["horizon_min"] == HORIZON)
          & (w["label"] != "CENSORED_IN_HORIZON") & (~w["movie_id"].isin(EXCLUDED_MOVIES))]
    y = np.select([w["label"] == "APOPTOSIS", w["label"] == "LYTIC"], [1, 2], default=0)
    out = w[["movie_id", "track_id", "t_obs_frame", "group", "window_weight"]].copy()
    out["y"] = y
    return out


def inner_validation_group(labels, training_groups) -> str:
    """The training group with the most lytic events; never the held-out group."""
    lytic = labels[(labels["y"] == 2) & (labels["group"].isin(training_groups))]
    if lytic.empty:
        raise SystemExit("no lytic events in any training group; the selection group is undefined")
    counts = lytic.groupby("group").size().sort_values(ascending=False)
    return str(counts.index[0])


def cell_tag(lr: float, ld: float) -> str:
    """File-name tag of one setting, e.g. ``lr0.0002_ld0.9``."""
    return f"lr{lr:g}_ld{ld:g}"


def checkpoint_path(runs_dir: Path, fold: str, lr: float, ld: float, seed: int, *, final: bool) -> Path:
    """``ft__<fold>__<cell>[__final]__s<seed>.pt`` under ``runs_dir``."""
    tag = f"{fold}__{cell_tag(lr, ld)}" + ("__final" if final else "")
    return runs_dir / f"ft__{tag}__s{seed}.pt"


def class_weights(counts: dict) -> np.ndarray:
    """Inverse-frequency class weights ``sum(n) / (3 * n_c)`` in ``CLASSES`` order."""
    freq = np.array([max(1, counts[c]) for c in CLASSES], dtype=np.float64)
    return freq.sum() / (3 * freq)


def train(model, data, store, mean, std, dev, *, steps: int, split, seed: int, cls_w, state: Path, run: dict,
          backbone_lr: float, layer_decay: float, ckpt_every: int = 500, log=print):
    """Run the fine-tuning loop; returns the per-step losses, or None if interrupted (state saved).

    ``model`` is the classifier on ``dev``; ``data`` holds the clip rows with ``y`` and ``window_weight``;
    ``run`` identifies the run in its state file. The loss is class-weighted cross-entropy times the window
    weight, normalised by the summed window weight of the 128 rows, so accumulation equals one batch of 128.
    """
    import torch
    import torch.nn.functional as Fn

    from . import training as T
    from .model import layer_decay_groups

    n_enc = model.config.num_hidden_layers
    groups = layer_decay_groups(model.named_parameters(), base_lr=backbone_lr, head_lr=HEAD_LR,
                               decay=layer_decay, weight_decay=WEIGHT_DECAY, n_layers=n_enc)
    opt = torch.optim.AdamW(groups)
    sched = T.warmup_cosine(opt, steps, WARMUP_FRACTION)
    cls_w = torch.tensor(cls_w, dtype=torch.float32, device=dev)
    start, rng = 0, np.random.default_rng(seed)
    g_aug = torch.Generator(device=dev).manual_seed(seed + T.AUG_STREAM_OFFSET)
    K = T.blur_kernel(dev)
    losses: list[float] = []
    if state.exists():
        start, rng, losses = T.load_state(state, model, opt, sched, g_aug, dev, run=run)
        log(f"  resumed from step {start} of {steps}")

    gather = ClipGatherer(store)
    movies = data["movie_id"].to_numpy(); rows = data["frame_rows"].to_numpy()
    ys = data["y"].to_numpy(); ws = data["window_weight"].to_numpy().astype(np.float32)
    stop = T.Interrupt()
    log_every = max(1, min(50, steps // 4))
    t0 = T.now()
    for step in range(start, steps):
        opt.zero_grad(set_to_none=True)
        # The 128 rows are drawn first and the loss is normalised by their summed weight.
        pick = rng.integers(0, len(data), size=split.effective)
        wsum = float(ws[pick].sum()) or 1e-6
        step_loss, step_finite = 0.0, True
        for c in range(split.grad_accum):
            sl = pick[c * split.micro_batch:(c + 1) * split.micro_batch]
            x = T.to_model_input(T.clip_batch(gather, movies, rows, sl, dev, g_aug, K), mean, std)
            y = torch.from_numpy(ys[sl]).long().to(dev)
            w = torch.from_numpy(ws[sl]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(pixel_values=x).logits
            ce = Fn.cross_entropy(logits.float(), y, weight=cls_w, reduction="none")
            loss = (ce * w).sum() / wsum
            finite = bool(torch.isfinite(loss))
            step_finite = step_finite and finite
            step_loss += float(loss.detach())
            loss.backward()                  # no loss scaler
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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


def training_rows(idx, labels, fold: str, *, final: bool):
    """Labelled clip rows to train on and the selection group.

    Grid runs leave out the selection group; ``final`` runs use all eight training groups.
    """
    if fold in set(idx["group"]) or idx["movie_id"].astype(str).str.startswith(f"{fold}_").any():
        raise ClipBuildError(f"the clip index of fold {fold} holds rows of the held-out group")
    train_groups = sorted(set(idx["group"]))
    inner = inner_validation_group(labels, train_groups)
    if inner == fold:
        raise ClipBuildError("the selection group may not be the held-out group")
    use_groups = train_groups if final else [g for g in train_groups if g != inner]
    data = idx.merge(labels, on=["movie_id", "track_id", "t_obs_frame", "group"], how="inner")
    data = data[data["group"].isin(use_groups)].reset_index(drop=True)
    if data.empty:
        raise ClipBuildError("no labelled clips after joining the clip index and the labels")
    return data, inner, use_groups


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune VideoMAE-large end to end on one fold.")
    ap.add_argument("--fold", required=True, help="held-out group")
    ap.add_argument("--backbone-lr", type=float, required=True, choices=[5e-5, 2e-4])
    ap.add_argument("--layer-decay", type=float, required=True, choices=[0.75, 0.9])
    ap.add_argument("--final", action="store_true", help="train on all eight training groups")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--micro-batch", type=int, default=MICRO_BATCH, help="rows per forward pass; must divide 128")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--snapshot", type=Path, default=None,
                    help="local directory of the checkpoint (default: download the pinned revision)")
    ap.add_argument("--adapt-dir", type=Path, default=None)
    ap.add_argument("--clips-dir", type=Path, default=None)
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--windows", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    import pandas as pd
    import torch

    from . import training as T
    from .model import (build_classifier, checkpoint_tensors, image_stats, load_adapted_encoder, pretrained_snapshot,
                        resolve_device)

    adapt_dir = a.adapt_dir or default_adapt_dir()
    clips_dir = a.clips_dir or default_clips_dir()
    store = a.store or default_store()
    windows = a.windows or default_windows()
    out_dir = a.out_dir or default_out_dir()
    split = T.split_batch(a.micro_batch)
    ckpt = checkpoint_path(out_dir, a.fold, a.backbone_lr, a.layer_decay, a.seed, final=a.final)
    adapted = adapt_dir / f"adapt__{a.fold}.pt"
    run = {"steps": a.steps, "seed": a.seed, "effective": split.effective}
    if adapted.is_file():
        run["adapted_sha256"] = T.sha256_file(adapted)
    if ckpt.is_file():
        return T.finished_run(ckpt, run)
    clip_index = clips_dir / f"clips__{a.fold}.parquet"
    if (input_missing(adapted, "video-adapt") or input_missing(clip_index, "video-clips")
            or input_missing(windows, "data-windows")):
        return 3

    idx = pd.read_parquet(clip_index)
    labels = label_table(windows)
    data, inner, use_groups = training_rows(idx, labels, a.fold, final=a.final)
    counts = {CLASSES[i]: int((data["y"] == i).sum()) for i in range(3)}
    print(f"fine-tune fold {a.fold}, {cell_tag(a.backbone_lr, a.layer_decay)}, seed {a.seed}"
          f"{' (final)' if a.final else ''}: selection group {inner}; {len(data)} rows {counts}; "
          f"{a.steps} steps of {split.micro_batch} x {split.grad_accum}", flush=True)

    snap = a.snapshot or pretrained_snapshot()
    mean, std = image_stats(snap)
    ck = checkpoint_tensors(snap)
    dev = resolve_device(a.device)
    torch.manual_seed(a.seed)                  # before the model and its new head are built
    model = build_classifier(snap, ck)         # attention biases restored on the CPU, then moved
    model = model.to(dev)
    kept = load_adapted_encoder(model, adapted)
    print(f"  adapted encoder: {kept} tensors loaded and checked equal", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    state = ckpt.with_name(ckpt.name[:-len(".pt")] + ".state.pt")
    losses = train(model, data, store, mean, std, dev, steps=a.steps, split=split, seed=a.seed,
                   cls_w=class_weights(counts), state=state, run=run, backbone_lr=a.backbone_lr,
                   layer_decay=a.layer_decay, ckpt_every=a.ckpt_every, log=lambda s: print(s, flush=True))
    if losses is None:
        return 10
    # the record comes first, so a checkpoint never exists without it
    ckpt.with_suffix(".json").write_text(json.dumps(
        {"fold": a.fold, "backbone_lr": a.backbone_lr, "layer_decay": a.layer_decay, "final": a.final,
         **run, "micro_batch": split.micro_batch, "grad_accum": split.grad_accum,
         "selection_group": inner, "training_groups": use_groups, "class_counts": counts,
         "all_steps_finite": bool(np.isfinite(losses).all())}, indent=2) + "\n")
    T.atomic_save({"state_dict": model.state_dict()}, ckpt)
    print(f"wrote {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
