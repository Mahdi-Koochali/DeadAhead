"""Short fine-tune of VideoMAE-large on yeast death within 30 min, from one of two starts, then the held-out pass.

    deadahead yeast-finetune --start videomae_finetuned --heldout Pos3_3 --seed 2601562
    deadahead yeast-finetune --start videomae_public --heldout Pos3_3 --kind thr
    deadahead yeast-finetune --start videomae_finetuned --heldout Pos3_3 --kind le

Runs: ``final`` trains on the 6 training positions, predicts the held-out position and embeds every position;
``thr`` (seed 2601562) trains on the 5 positions other than the threshold position, predicts the threshold position
and embeds those 6; ``le`` (seed 2601562) trains on 3 of the 6 training positions and predicts the held-out position.

Recipe: 1,000 steps x effective batch 128; backbone lr 5e-5, layer decay 0.9, head lr 1e-3; AdamW weight decay 0.05;
5% warm-up then cosine; gradient clipping at global norm 1.0; bf16 autocast on CUDA, no loss scaler; class-weighted
cross-entropy with N / (2 N_c) from the run's own rows; the augmentation of ``deadahead.video.training``; a new linear
1,024 -> 2 head drawn after the seed is set; the weights at the last step are scored. The pass reads no label.
Resumable: ``weights/<run>.state.pt`` every 250 steps and at SIGTERM / SIGUSR1 (exit 10), removed once the weights are
saved. A state, weights or predictions of the same name from another --steps or effective batch are refused.
Outputs under ``runs/yeast``: the trained weights (``weights/<run>.pt``), a summary (``weights/<run>.json``), the
embeddings (``embed/<run>.npy`` and ``.keys.parquet``) and the predictions (``pred/<run>.parquet``: unit, position, t,
p_none, p_death). The predictions come last and only when every value is finite (otherwise exit 5), so their file marks
a finished run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from deadahead import paths

from . import constants as C
from . import data as D

CKPT_EVERY = 250
MICRO_BATCH = 16       # 16 x 8 = 128; 32 x 4 fits on GPUs with more than 80 GB
KINDS = ("final", "thr", "le")


def require_torch(ap: argparse.ArgumentParser) -> None:
    """Stop with the install command when the video dependencies are missing."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        ap.error('PyTorch is not installed: pip install -e ".[video]"')


def default_rows_dir() -> Path:
    return D.yeast_dir() / "rows"


def default_store() -> Path:
    return D.yeast_dir() / "vid_store"


def default_out_dir() -> Path:
    return paths.runs_dir() / "yeast"


def default_config() -> Path:
    return paths.REPO_ROOT / "configs" / "yeast.yaml"


def read_inputs(config: Path) -> tuple[dict, dict]:
    """The per-fold inputs (``folds``: channel, m_D, s_D per held-out position) and the source moments
    (``source_moments``: m_F, s_F) from a YAML or JSON file."""
    import yaml
    cfg = yaml.safe_load(Path(config).read_text())
    return cfg["folds"], cfg["source_moments"]


def run_name(kind: str, start: str, heldout: str, seed: int) -> str:
    """``<start>__<position>__s<seed>`` for a final run, ``<kind>__<start>__<position>`` for a thr or le run."""
    return f"{start}__{heldout}__s{seed}" if kind == "final" else f"{kind}__{start}__{heldout}"


def plan_of(kind: str, start: str, heldout: str, seed: int = C.SEED) -> dict:
    """What a run trains on and what its pass writes (pure)."""
    if kind not in KINDS or start not in C.STARTS:
        raise ValueError(f"unknown run kind {kind!r} or start {start!r}")
    if kind != "final" and seed != C.SEED:
        raise ValueError(f"the {kind} runs use seed {C.SEED}")
    train = D.row_file_positions(kind, heldout)
    if kind == "final":
        predict, embed = (heldout,), C.POSITIONS
    elif kind == "thr":
        predict, embed = (C.threshold_position(heldout),), tuple(train) + (C.threshold_position(heldout),)
    else:
        predict, embed = (heldout,), ()
    if heldout in train or (kind == "thr" and C.threshold_position(heldout) in train):
        raise ValueError(f"{kind} run of {heldout}: a withheld position is among the training positions")
    return {"kind": kind, "start": start, "heldout": heldout, "seed": seed, "row_kind": kind,
            "train_positions": tuple(train), "predict_positions": predict, "embed_positions": tuple(embed)}


def class_weights(data, dev):
    """Inverse-frequency weights N / (2 N_c) over the run's rows, and the class counts."""
    import torch
    counts = np.array([max(1, int((data["y"] == i).sum())) for i in range(len(C.CLASSES))], dtype=np.float64)
    return torch.tensor(counts.sum() / (len(C.CLASSES) * counts), dtype=torch.float32, device=dev), counts


def make_optimiser(model, steps: int):
    """AdamW over the layer-decay groups, linear warm-up then cosine."""
    import torch

    from deadahead.video import training as T
    from deadahead.video.model import layer_decay_groups
    groups = layer_decay_groups(model.named_parameters(), base_lr=C.BACKBONE_LR, head_lr=C.HEAD_LR,
                                decay=C.LAYER_DECAY, weight_decay=C.WEIGHT_DECAY,
                                n_layers=model.config.num_hidden_layers)
    opt = torch.optim.AdamW(groups)
    sched = T.warmup_cosine(opt, steps, C.WARMUP_FRACTION)
    return opt, sched, len(groups)


def train_steps(model, opt, sched, data, store, fold, src_m, mean, std, dev, split, cls_w, rng, g_aug, K, losses,
                start, stop_at, on_step=None):
    """Steps ``start`` .. ``stop_at - 1``: 128 rows drawn per step, micro-batches accumulated, clipped, stepped.
    The per-step loss is appended to ``losses``; ``on_step(next_step)`` may return an exit code to stop early."""
    import torch
    import torch.nn.functional as Fn

    from deadahead.video import training as T

    from . import model as M
    traps = data["unit"].to_numpy()
    frames = [D.clip_frames(t) for t in data["t"].to_numpy()]
    ys = data["y"].to_numpy()
    model.train()
    for step in range(start, stop_at):
        opt.zero_grad(set_to_none=True)
        pick = rng.integers(0, len(data), size=split.effective)       # the 128 rows are drawn first
        step_loss, step_finite = 0.0, True
        wsum = float(split.effective)
        for c in range(split.grad_accum):
            sl = pick[c * split.micro_batch:(c + 1) * split.micro_batch]
            x = M.mapped_batch(store, traps[sl], [frames[i] for i in sl], fold, src_m, dev)
            x = T.to_model_input(T.augment_clip(x, g_aug, K), mean, std)
            y = torch.from_numpy(ys[sl]).long().to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits, _ = M.logits_and_embedding(model, x)
            loss = Fn.cross_entropy(logits.float(), y, weight=cls_w, reduction="none").sum() / wsum
            fin = bool(torch.isfinite(loss))
            step_finite = step_finite and fin
            step_loss += float(loss.detach())
            loss.backward()                                            # no loss scaler
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        opt.step(); sched.step()
        losses.append(step_loss)
        if on_step is not None:
            rc = on_step(step + 1)
            if rc is not None:
                return rc
    return None


def train(model, data, store, fold, src_m, mean, std, dev, *, steps: int, split, seed: int, state: Path, run: dict,
          ckpt_every: int = CKPT_EVERY, log=print):
    """The fine-tuning loop with resumable state; returns the per-step losses, or None if interrupted.
    ``run`` identifies the run in its state file."""
    import torch

    from deadahead.video import training as T
    opt, sched, _ = make_optimiser(model, steps)
    cls_w, _ = class_weights(data, dev)
    start, rng = 0, np.random.default_rng(seed)
    g_aug = torch.Generator(device=dev).manual_seed(seed + T.AUG_STREAM_OFFSET)
    K = T.blur_kernel(dev)
    losses: list[float] = []
    if state.exists():
        start, rng, losses = T.load_state(state, model, opt, sched, g_aug, dev, run=run)
        log(f"  resumed from step {start} of {steps}")
    stop = T.Interrupt()

    def on_step(s_next: int):
        if ckpt_every and s_next % ckpt_every == 0 and s_next < steps:
            T.save_state(state, model, opt, sched, s_next, rng, g_aug, losses, run=run)
        if stop.hit:
            T.save_state(state, model, opt, sched, s_next, rng, g_aug, losses, run=run)
            log(f"  interrupted at step {s_next}; state saved, re-run to resume")
            return 10
        return None

    rc = train_steps(model, opt, sched, data, store, fold, src_m, mean, std, dev, split, cls_w, rng, g_aug, K,
                     losses, start, steps, on_step)
    if rc is not None:
        return None
    if ckpt_every and steps > start:
        T.save_state(state, model, opt, sched, steps, rng, g_aug, losses, run=run)
    return losses


def write_pass(model, plan: dict, all_keys, store, fold, src_m, mean, std, dev, out_dir: Path, name: str) -> dict:
    """The pass without labels: embeddings on the embed positions, then predictions on the predict positions, skipped
    when a value is not finite."""
    from . import model as M
    keys = all_keys[all_keys["position"].isin(plan["embed_positions"] or plan["predict_positions"])]
    keys = keys.reset_index(drop=True)
    probs, emb = M.predict(model, store, keys, fold, src_m, mean, std, dev,
                           want_embedding=bool(plan["embed_positions"]))
    finite = bool(np.isfinite(probs).all()) and (emb is None or bool(np.isfinite(emb).all()))
    pred = keys.assign(p_none=probs[:, 0], p_death=probs[:, 1])
    pred = pred[pred["position"].isin(plan["predict_positions"])].reset_index(drop=True)
    pf = out_dir / "pred" / f"{name}.parquet"
    out = {"predictions": str(pf), "probs_finite": finite}
    if emb is not None:
        ef = out_dir / "embed" / f"{name}.npy"
        ef.parent.mkdir(parents=True, exist_ok=True)
        tmp = ef.with_name(ef.name + f".tmp{os.getpid()}.npy")
        np.save(tmp, emb)
        os.replace(tmp, ef)
        keys.to_parquet(ef.with_suffix(".keys.parquet"), index=False)
        out["embeddings"] = str(ef)
    if finite:                                                     # last: main takes this file as the end of a run
        pf.parent.mkdir(parents=True, exist_ok=True)
        tmp = pf.with_name(pf.name + f".tmp{os.getpid()}")
        pred.to_parquet(tmp, index=False)
        os.replace(tmp, pf)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Short fine-tune of VideoMAE-large on yeast death, then the pass.")
    ap.add_argument("--start", required=True, choices=list(C.STARTS))
    ap.add_argument("--heldout", required=True, choices=list(C.POSITIONS), help="held-out position")
    ap.add_argument("--kind", default="final", choices=list(KINDS))
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--steps", type=int, default=C.STEPS)
    ap.add_argument("--micro-batch", type=int, default=MICRO_BATCH, help="rows per forward pass; must divide 128")
    ap.add_argument("--ckpt-every", type=int, default=CKPT_EVERY)
    ap.add_argument("--snapshot", type=Path, default=None,
                    help="local directory of the public checkpoint (default: download the pinned revision)")
    ap.add_argument("--start-weights", type=Path, default=None,
                    help="released fine-tuned VideoMAE for the videomae_finetuned start "
                         "(default: weights/videomae_large_finetuned__PNT1A_bp__seed<seed>.safetensors)")
    ap.add_argument("--config", type=Path, default=None, help="per-fold inputs (default: configs/yeast.yaml)")
    ap.add_argument("--rows-dir", type=Path, default=None)
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    try:
        plan = plan_of(a.kind, a.start, a.heldout, a.seed)
    except ValueError as e:
        ap.error(str(e))
    require_torch(ap)

    import pandas as pd
    import torch

    from deadahead.video import training as T
    from deadahead.video.clips import input_missing
    from deadahead.video.model import image_stats, pretrained_snapshot

    from . import model as M

    name = run_name(a.kind, a.start, a.heldout, a.seed)
    out_dir = a.out_dir or default_out_dir()
    rows_dir = a.rows_dir or default_rows_dir()
    pf = out_dir / "pred" / f"{name}.parquet"
    wdir = out_dir / "weights"
    weights, state = wdir / f"{name}.pt", wdir / f"{name}.state.pt"
    split = T.split_batch(a.micro_batch)
    run = {"steps": a.steps, "seed": a.seed, "effective": split.effective}
    if (pf.is_file() or weights.is_file()) and T.run_differs(weights, run, also=pf):
        return 3
    if pf.is_file():
        print(f"{pf} exists; nothing to do")
        return 0
    if input_missing(rows_dir / "all_keys.parquet", "yeast-data"):
        return 3
    folds, src_m = read_inputs(a.config or default_config())
    fold = folds[a.heldout]
    data = D.read_row_file(rows_dir, plan["row_kind"], a.heldout)
    all_keys = pd.read_parquet(rows_dir / "all_keys.parquet")
    store = D.VidStore(a.store or default_store())
    snap = a.snapshot or pretrained_snapshot()
    mean, std = image_stats(snap)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    wdir.mkdir(parents=True, exist_ok=True)

    if weights.is_file():                                          # trained earlier: only the held-out pass is left
        model = M.load_yeast_classifier(weights, snap, dev)
        losses = None
        print(f"{weights} exists; running the pass only", flush=True)
    else:
        start_weights = None
        if a.start == "videomae_finetuned":
            start_weights = a.start_weights or M.released_weights(a.seed)
            if not Path(start_weights).is_file():
                print(f"{start_weights} not found: the released weights are listed in docs/WEIGHTS.md", file=sys.stderr)
                return 3
        model, notes = M.build_model(a.start, a.seed, snap, dev, start_weights=start_weights)
        print(f"yeast fine-tune {name}: train on {', '.join(plan['train_positions'])}; {len(data)} rows, "
              f"{int(data['y'].sum())} deaths within 30 min; {a.steps} steps of {split.micro_batch} x "
              f"{split.grad_accum}; {notes['init']}", flush=True)
        losses = train(model, data, store, fold, src_m, mean, std, dev, steps=a.steps, split=split, seed=a.seed,
                       state=state, run=run, ckpt_every=a.ckpt_every, log=lambda s: print(s, flush=True))
        if losses is None:
            return 10
        _, counts = class_weights(data, dev)
        # the record comes first, so the weights never exist without it
        weights.with_suffix(".json").write_text(json.dumps(
            {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in plan.items()}, **run,
             "micro_batch": split.micro_batch, "grad_accum": split.grad_accum, "rows": int(len(data)),
             "class_counts": counts.astype(int).tolist(), "channel": fold["channel"],
             "all_steps_finite": bool(np.isfinite(losses).all()), "losses": [float(v) for v in losses]},
            indent=1) + "\n")
        T.atomic_save({"state_dict": model.state_dict()}, weights)
    state.unlink(missing_ok=True)
    res = write_pass(model, plan, all_keys, store, fold, src_m, mean, std, dev, out_dir, name)
    if not res["probs_finite"]:
        print("some probabilities are not finite")
        return 5
    print(f"wrote {res['predictions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
