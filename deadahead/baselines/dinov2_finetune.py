"""Fine-tune DINOv2 ViT-B/14 on the 30-minute death labels of the training groups of one fold.

The training rows are the landmarks of the primary panel outside the held-out group whose 30-min outcome is known in
the complete movie, each with its crop at the landmark frame. The last two transformer blocks and the final norm are
tuned with a linear head on three classes (no death, apoptosis, lytic death): cross-entropy with class weights
sum(n) / (3 n_c), weighted by the window weight, under the augmentation and budget of the adaptation (6,000 steps
of 128 crops, AdamW lr 1e-5, weight decay 0.05). It starts from the pretrained encoder.

    deadahead baseline-dinov2-finetune --fold DU145_bp
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from deadahead.survival.config import head_config, windows_config
from deadahead.survival.frames import HORIZONS_MIN, load_tables
from deadahead.survival.oracle import build_oracle

SEED = 2601562
HORIZON = 30
STEPS = 6000
BATCH = 128
BLOCKS = 2
LR = 1e-5
RES = 224
CLASSES = ("NONE", "APOPTOSIS", "LYTIC")


def build_labelled_set(group: str, horizon: int, windows: pd.DataFrame, events: pd.DataFrame, movies: pd.DataFrame,
                       cfg: dict, cfg_w: dict, crops: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Crops, labels (0 none, 1 apoptosis, 2 lytic) and window weights of every landmark outside ``group`` whose
    outcome at ``horizon`` is known in the complete movie; a landmark of unknown outcome is left out."""
    windows = windows[windows["panel"] == cfg["protocol"]["panel"]]
    windows = windows[~windows["movie_id"].isin(cfg["protocol"]["morphology_excluded_movies"])]

    oracle = build_oracle(windows, events, movies, cfg_w)
    lm = oracle[(oracle["horizon_min"] == horizon) & (oracle["group"] != group)].copy()
    if lm.empty:
        raise SystemExit(f"FAIL: no training landmarks left after excluding {group}")

    # the crop store is keyed by cell number
    cn = windows[["movie_id", "track_id"]].drop_duplicates()
    cn["cell_num"] = cn["track_id"].str.rsplit(":", n=1).str[-1].astype(int)
    lm = lm.merge(cn, on=["movie_id", "track_id"], how="left")

    X, Y, W, miss = [], [], [], 0
    for m in sorted(lm["movie_id"].unique()):
        sub = lm[lm["movie_id"] == m]
        idx = pd.read_parquet(crops / f"{m}.parquet")
        row = {(int(c), int(f)): int(r)
               for c, f, r in zip(idx["cell_num"], idx["frame_index"], idx["row"])}
        arr = np.load(crops / f"{m}.npy", mmap_mode="r")
        take, keep = [], []
        for i, rec in enumerate(sub.itertuples(index=False)):
            r = row.get((int(rec.cell_num), int(rec.t_obs_frame)))
            if r is None:
                miss += 1
                continue
            take.append(r); keep.append(i)
        if not take:
            continue
        X.append(np.asarray(arr[np.array(take)], dtype=np.float16))
        Y.append(sub["oracle_label"].to_numpy()[keep])
        W.append(sub["window_weight"].to_numpy()[keep])
    X = np.concatenate(X); Y = np.concatenate(Y).astype(np.int64); W = np.concatenate(W)
    card = {"n_landmarks": int(len(X)), "n_crops_missing_from_store": int(miss),
            "n_movies": int(lm["movie_id"].nunique()),
            "class_counts": {k: int((Y == v).sum()) for k, v in
                             (("NONE", 0), ("APOPTOSIS", 1), ("LYTIC", 2))},
            "held_out_group": group, "groups_trained_on": sorted(lm["group"].unique().tolist())}
    return X, Y, W, card


def finetune_encoder(Xn: np.ndarray, Yn: np.ndarray, Wn: np.ndarray, *, steps: int = STEPS, batch: int = BATCH,
                     blocks: int = BLOCKS, lr: float = LR, res: int = RES, seed: int = SEED, dev=None,
                     load_model=None) -> dict:
    """Tune the last ``blocks`` blocks, the final norm and a 3-class linear head on crops ``Xn`` (raw phase), labels
    ``Yn`` and window weights ``Wn``.

    ``load_model(dev)`` builds the pretrained encoder (default: DINOv2 ViT-B/14 from the pinned hub commit), after the
    global torch seed is set. Returns the tuned encoder and head, the per-step losses and accuracies, the class
    weights and the tuned and total parameter counts.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    from . import dinov2_common as C

    dev = dev or C.default_device()
    load_model = load_model or (lambda d: C.load_encoder(C.MODEL_NAME, d))
    X = torch.from_numpy(Xn).to(dev)
    Y = torch.from_numpy(Yn).to(dev)
    W = torch.from_numpy(Wn.astype(np.float32)).to(dev)
    # inverse-frequency class weights: deaths are about 3 % of the rows
    cnt = torch.bincount(Y, minlength=3).float().clamp_min(1.0)
    cls_w = (cnt.sum() / (3.0 * cnt))
    print(f"  class weights {[round(float(v), 3) for v in cls_w]}", flush=True)

    torch.manual_seed(seed)
    g = torch.Generator(device=dev); g.manual_seed(seed)
    gi = torch.Generator(device=dev); gi.manual_seed(seed + 1)
    model = load_model(dev).train()
    for p in model.parameters():
        p.requires_grad_(False)
    tuned = []
    for blk in list(getattr(model, "blocks", []))[-blocks:]:
        for p in blk.parameters():
            p.requires_grad_(True); tuned.append(p)
    if hasattr(model, "norm"):
        for p in model.norm.parameters():
            p.requires_grad_(True); tuned.append(p)
    head = nn.Linear(768, 3).to(dev)
    n_tuned = sum(p.numel() for p in tuned)
    n_total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(list(tuned) + list(head.parameters()), lr=lr, weight_decay=0.05)

    bf16 = dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    mean, std = C.IMAGENET_MEAN.to(dev), C.IMAGENET_STD.to(dev)
    K = C._blur_kernel(dev)
    lo, hi = C.GLOBAL_LO, C.GLOBAL_HI
    losses, accs, nb = [], [], min(batch, len(X))
    print(f"  autocast {amp_dtype} on "
          f"{torch.cuda.get_device_name(0) if dev.type=='cuda' else 'cpu'}", flush=True)
    t0 = time.time()

    for step in range(steps):
        idx = torch.randperm(len(X), generator=gi, device=dev)[:nb]
        # the global phase map of the embedding
        x = X[idx].float()
        x = torch.log1p(9.0 * ((x - lo) / (hi - lo)).clamp(0, 1)) / float(np.log(10.0))
        x = x.unsqueeze(1)
        y, w = Y[idx], W[idx]
        with torch.autocast("cuda", enabled=dev.type == "cuda", dtype=amp_dtype):
            v = C.augment(x, g, K).repeat(1, 3, 1, 1)
            v = Fn.interpolate(v, size=(res, res), mode="bilinear", align_corners=False)
            v = (v - mean) / std
            f = model(v)
            if not torch.is_tensor(f):
                po = getattr(f, "pooler_output", None)
                f = po if po is not None else f.last_hidden_state[:, 0]
            logits = head(f.float())
            ce = Fn.cross_entropy(logits, y, weight=cls_w, reduction="none")
            loss = (ce * w).sum() / w.sum().clamp_min(1e-6)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        losses.append(float(loss))
        accs.append(float((logits.argmax(1) == y).float().mean()))
        if step % 100 == 0:
            print(f"  step {step:>4}  loss {np.mean(losses[-100:]):.4f}  "
                  f"acc {np.mean(accs[-100:]):.3f}  [{(time.time()-t0)/60:.1f} min]", flush=True)
    return {"model": model, "head": head, "losses": losses, "accs": accs, "n_tuned": int(n_tuned),
            "n_total": int(n_total), "class_weights": [float(v) for v in cls_w], "amp_dtype": str(amp_dtype)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune DINOv2 ViT-B/14 on the 30-min death labels of one fold.")
    ap.add_argument("--fold", required=True, help="held-out group; it contributes no crop and no label")
    ap.add_argument("--horizon", type=int, default=HORIZON, choices=HORIZONS_MIN)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--blocks", type=int, default=BLOCKS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--res", type=int, default=RES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tables", type=Path, default=None, help="folder of windows, events and movies (default: tables/)")
    ap.add_argument("--crops", type=Path, default=None, help="crop store (default: crops/frames128)")
    ap.add_argument("--hub-repo-dir", type=Path, default=None,
                    help="local copy of the pinned facebookresearch/dinov2 commit (default: download it)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: runs/dinov2_finetuned/finetune/finetuned__<fold>.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    import torch

    from . import dinov2_common as C

    ckpt = a.out or C.finetuned_checkpoint(a.fold)
    if ckpt.exists() and not a.force:
        meta = torch.load(ckpt, map_location="cpu", weights_only=False).get("recipe", {})
        if meta.get("steps") == a.steps and meta.get("batch") == a.batch:
            print(f"{ckpt} exists ({a.steps} steps x batch {a.batch}); nothing to do")
            return 0
    t0 = time.time()
    tb = load_tables(a.tables, names=("windows", "events", "movies"))
    if a.fold not in set(tb["windows"]["group"]):
        ap.error(f"--fold: no group {a.fold!r} in the windows table")
    Xn, Yn, Wn, card = build_labelled_set(a.fold, a.horizon, tb["windows"], tb["events"], tb["movies"],
                                          head_config(), windows_config(), a.crops or C.crops_store())
    if a.fold in card["groups_trained_on"]:
        raise AssertionError("the held-out group is in the training set")
    print(f"{a.fold}: {card['n_landmarks']:,} landmarks from {card['n_movies']} movies, classes "
          f"{card['class_counts']}, {card['n_crops_missing_from_store']} without a crop", flush=True)
    dev = C.default_device(a.device)
    out = finetune_encoder(Xn, Yn, Wn, steps=a.steps, batch=a.batch, blocks=a.blocks, lr=a.lr, res=a.res,
                           seed=a.seed, dev=dev,
                           load_model=lambda d: C.load_encoder(C.MODEL_NAME, d, a.hub_repo_dir))
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt.with_name(ckpt.name + ".tmp")
    torch.save({"state_dict": dict(out["model"].state_dict()),
                "recipe": {"steps": a.steps, "batch": a.batch, "lr": a.lr, "res": a.res, "seed": a.seed,
                           "blocks": a.blocks, "fold": a.fold, "horizon_min": a.horizon,
                           "objective": "weighted_cross_entropy_3_class", "backbone": C.MODEL_NAME,
                           "hub_commit": C.HUB_COMMIT, "class_weights": out["class_weights"],
                           "autocast_dtype": out["amp_dtype"], **card}}, tmp)
    tmp.replace(ckpt)
    ls = out["losses"]
    print(f"{a.fold}: loss {np.mean(ls[:100]):.4f} -> {np.mean(ls[-100:]):.4f}, {out['n_tuned'] / 1e6:.1f}M of "
          f"{out['n_total'] / 1e6:.1f}M parameters tuned, {(time.time() - t0) / 60:.1f} min -> {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
