"""Adapt DINOv2 ViT-B/14 to the phase images of the training groups of one fold, without labels.

Step 1 draws a pool of random image tiles from the movies of the eight training groups (the held-out group
contributes no pixel): 15 evenly spaced frames per movie, square tiles of 96 to 259 px at random positions,
the global phase map, resized to 128 px; about 60,000 tiles per fold. Step 2 tunes the last two transformer
blocks and the final norm of the pretrained encoder with the NT-Xent (SimCLR) objective on two augmented views
per tile (6,000 steps of 128 tiles, AdamW lr 1e-5, weight decay 0.05). The rest of the encoder is not updated.

    deadahead baseline-dinov2-adapt --fold DU145_bp
    deadahead baseline-dinov2-adapt --fold DU145_bp --pool-only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from deadahead import paths

SIZE, SEED = 128, 2601562
GLOBAL_LO, GLOBAL_HI = -0.3, 4.0
TARGET = 60_000                 # tiles per pool, shared evenly over the training movies

#: Recipe of the adaptation.
STEPS = 6000
BATCH = 128
BLOCKS = 2                      # last transformer blocks tuned, with the final norm
LR = 1e-5
RES = 224


def default_pool(fold: str) -> Path:
    return paths.runs_dir() / "dinov2" / "pools" / f"{fold}.npy"


# ------------------------------------------------------------------ the tile pool
def _tile(img, rng, n, lo=96, hi=260):
    """``n`` square tiles of random side in [lo, min(hi, H, W)) at random positions of ``img``."""
    H, W = img.shape
    out = []
    for _ in range(n):
        sz = int(rng.integers(lo, min(hi, min(H, W))))
        y = int(rng.integers(0, max(H - sz, 1))); x = int(rng.integers(0, max(W - sz, 1)))
        out.append(img[y:y + sz, x:x + sz])
    return out


def _resize01(c):
    """A tile in [0, 1] resized to SIZE x SIZE through 8-bit bilinear resampling, as float16 in [0, 1]."""
    im = Image.fromarray((np.clip(c, 0, 1) * 255).astype(np.uint8)).resize((SIZE, SIZE),
                                                                          Image.BILINEAR)
    return np.asarray(im, np.float16) / 255.0


def movie_table(path: Path | None = None) -> pd.DataFrame:
    """The movie table (``tables/movies.csv``) sorted by movie, with its group and absolute image paths."""
    inv = pd.read_csv(path or paths.tables_dir() / "movies.csv").sort_values("movie_id")
    inv["group"] = inv["cell_line"] + "_" + inv["treatment"]
    inv["image_path"] = [str(p if Path(p).is_absolute() else paths.raw_dir() / p) for p in inv["image_path"]]
    return inv


def fold_pool(inv: pd.DataFrame, held: str) -> np.ndarray:
    """The tile pool of the fold that holds out group ``held``: (n, SIZE, SIZE) float16 in [0, 1]."""
    rng = np.random.default_rng(SEED)
    keep = inv[inv["group"] != held]
    per = max(1, TARGET // max(len(keep), 1)); frames, tiles = 15, max(1, per // 15)
    out = []
    for r in keep.itertuples():
        with tifffile.TiffFile(r.image_path) as tf:
            n = len(tf.pages)
            for f in np.linspace(0, n - 1, min(frames, n)).astype(int):
                img = tf.pages[int(f)].asarray().astype(np.float32)
                for c in _tile(img, rng, tiles):
                    lin = np.clip((c - GLOBAL_LO) / (GLOBAL_HI - GLOBAL_LO), 0, 1)
                    out.append(_resize01(np.log1p(9.0 * lin) / np.log(10.0)))
    return np.stack(out)


# ------------------------------------------------------------------ the adaptation
def adapt_encoder(pool: np.ndarray, *, steps: int = STEPS, batch: int = BATCH, blocks: int = BLOCKS,
                  lr: float = LR, res: int = RES, seed: int = SEED, dev=None, load_model=None) -> dict:
    """Tune the last ``blocks`` blocks and the final norm of the encoder on the tile pool with NT-Xent.

    ``load_model(dev)`` builds the pretrained encoder (default: DINOv2 ViT-B/14 from the pinned hub commit); it is
    called after the global torch seed is set, as its random initialisation draws from that generator. Returns the
    tuned encoder, the per-step losses and the tuned and total parameter counts.
    """
    import torch
    import torch.nn.functional as Fn

    from . import dinov2_common as C

    dev = dev or C.default_device()
    load_model = load_model or (lambda d: C.load_encoder(C.MODEL_NAME, d))
    augment, nt_xent, Head = C.augment, C.nt_xent, C.Head
    A = torch.from_numpy(pool).to(dev)
    torch.manual_seed(seed)
    g = torch.Generator(device=dev); g.manual_seed(seed)
    model = load_model(dev).train()

    for p in model.parameters():
        p.requires_grad_(False)
    tuned = []
    blocks_ = model.blocks if hasattr(model, "blocks") else []
    for blk in list(blocks_)[-blocks:]:
        for p in blk.parameters():
            p.requires_grad_(True); tuned.append(p)
    if hasattr(model, "norm"):
        for p in model.norm.parameters():
            p.requires_grad_(True); tuned.append(p)
    head = Head().to(dev)
    n_tuned = sum(p.numel() for p in tuned)
    n_total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(list(tuned) + list(head.parameters()), lr=lr, weight_decay=0.05)
    # bf16 autocast from compute capability 8.0, fp16 below it; the loss scaler is on for every CUDA run
    bf16 = dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"autocast dtype: {amp_dtype} on {torch.cuda.get_device_name(0) if dev.type=='cuda' else 'cpu'}",
          flush=True)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    gi = torch.Generator(device=dev); gi.manual_seed(seed + 1)     # batch draws, apart from the augmentations
    mean, std = C.IMAGENET_MEAN.to(dev), C.IMAGENET_STD.to(dev)
    K = C._blur_kernel(dev)
    losses, t0 = [], time.time()
    nb = min(batch, len(A))
    for step in range(steps):
        idx = torch.randperm(len(A), generator=gi, device=dev)[:nb]
        x = A[idx].float().unsqueeze(1).clamp(0, 1)
        with torch.autocast("cuda", enabled=dev.type == "cuda", dtype=amp_dtype):
            zs = []
            for _ in range(2):
                v = augment(x, g, K).repeat(1, 3, 1, 1)
                v = Fn.interpolate(v, size=(res, res), mode="bilinear", align_corners=False)
                v = (v - mean) / std
                f = model(v)
                if not torch.is_tensor(f):
                    po = getattr(f, "pooler_output", None)
                    f = po if po is not None else f.last_hidden_state[:, 0]
                zs.append(head(f.float()))
            loss = nt_xent(*zs)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        losses.append(float(loss))
        if step % 100 == 0:
            print(f"  step {step:>4}  loss {np.mean(losses[-100:]):.4f}  "
                  f"[{(time.time()-t0)/60:.1f} min]", flush=True)
    return {"model": model, "losses": losses, "n_tuned": int(n_tuned), "n_total": int(n_total),
            "amp_dtype": str(amp_dtype)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Adapt DINOv2 ViT-B/14 without labels to the training groups of one fold.")
    ap.add_argument("--fold", required=True, help="held-out group; its movies contribute no tile")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--blocks", type=int, default=BLOCKS, help="final transformer blocks to tune")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--res", type=int, default=RES, help="input resolution; a multiple of the patch size 14")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--movie-table", type=Path, default=None, help="default: tables/movies.csv")
    ap.add_argument("--pool", type=Path, default=None, help="tile pool file (default: runs/dinov2/pools/<fold>.npy)")
    ap.add_argument("--pool-only", action="store_true", help="build the tile pool and stop")
    ap.add_argument("--hub-repo-dir", type=Path, default=None,
                    help="local copy of the pinned facebookresearch/dinov2 commit (default: download it)")
    ap.add_argument("--out", type=Path, default=None, help="default: runs/dinov2/adapt/adapted__<fold>.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="retrain even if the checkpoint exists")
    a = ap.parse_args(argv)

    pool_path = a.pool or default_pool(a.fold)
    if not pool_path.is_file():
        inv = movie_table(a.movie_table)
        if a.fold not in set(inv["group"]):
            raise SystemExit(f"unknown group {a.fold}")
        A = fold_pool(inv, a.fold)
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = pool_path.with_name(pool_path.stem + ".tmp.npy")
        np.save(tmp, A)
        tmp.replace(pool_path)
        print(f"{a.fold}: pool {A.shape} from {int((inv['group'] != a.fold).sum())} movies -> {pool_path}", flush=True)
    if a.pool_only:
        return 0

    import torch

    from . import dinov2_common as C

    ckpt = a.out or C.adapted_checkpoint(a.fold)
    if ckpt.exists() and not a.force:
        meta = torch.load(ckpt, map_location="cpu", weights_only=False).get("recipe", {})
        if meta.get("steps") == a.steps and meta.get("batch") == a.batch:
            print(f"{ckpt} exists ({a.steps} steps x batch {a.batch}); nothing to do")
            return 0
    dev = C.default_device(a.device)
    A = np.load(pool_path)
    print(f"{a.fold}: pool {tuple(A.shape)} on {dev}", flush=True)
    out = adapt_encoder(A, steps=a.steps, batch=a.batch, blocks=a.blocks, lr=a.lr, res=a.res, seed=a.seed, dev=dev,
                        load_model=lambda d: C.load_encoder(C.MODEL_NAME, d, a.hub_repo_dir))
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt.with_name(ckpt.name + ".tmp")
    torch.save({"state_dict": {k: v for k, v in out["model"].state_dict().items()},
                "recipe": {"steps": a.steps, "batch": a.batch, "lr": a.lr, "res": a.res, "seed": a.seed,
                           "blocks": a.blocks, "fold": a.fold, "objective": "nt_xent",
                           "backbone": C.MODEL_NAME, "hub_commit": C.HUB_COMMIT,
                           "n_tiles": int(len(A)), "autocast_dtype": out["amp_dtype"]}}, tmp)
    tmp.replace(ckpt)
    ls = out["losses"]
    print(f"{a.fold}: loss {np.mean(ls[:100]):.4f} -> {np.mean(ls[-100:]):.4f}, "
          f"{out['n_tuned'] / 1e6:.1f}M of {out['n_total'] / 1e6:.1f}M parameters tuned -> {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
