"""DINOv2 ViT-B/14 pieces shared by the DINOv2 baselines: loading the encoder, the phase map, the augmentation, the
contrastive objective, and one forward pass over a crop store.

The encoder comes from ``torch.hub`` at a fixed commit of facebookresearch/dinov2 (``HUB_COMMIT``) with the
LVD-142M weights ``dinov2_vitb14_pretrain.pth``. The model's random initialisation draws from the global torch
generator before the pretrained weights are loaded, so the code version also fixes the initial weights of the
heads built after it.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from deadahead import paths

HUB_REPO = "facebookresearch/dinov2"
HUB_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
MODEL_NAME = "dinov2_vitb14"
EMBED_DIM = 768
#: sha256 of the LVD-142M checkpoint the hub entry point downloads (dl.fbaipublicfiles.com/dinov2)
WEIGHTS_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"

# The encoder was pretrained on ImageNet-normalised RGB images.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Fixed bounds of the global phase map (not fitted to any fold).
GLOBAL_LO, GLOBAL_HI = -0.3, 4.0


# ------------------------------------------------------------------ locations
def crops_store() -> Path:
    """The 128-px crop store read by the DINOv2 baselines: every 5th frame (``deadahead video-crops --stride 5
    --out-dir <data>/crops/frames128``)."""
    return paths.crops_dir() / "frames128"


def adapted_checkpoint(fold: str) -> Path:
    """The encoder adapted without labels for the held-out group ``fold``."""
    return paths.runs_dir() / "dinov2" / "adapt" / f"adapted__{fold}.pt"


def finetuned_checkpoint(fold: str) -> Path:
    """The encoder fine-tuned on the death labels of the groups other than ``fold``."""
    return paths.runs_dir() / "dinov2_finetuned" / "finetune" / f"finetuned__{fold}.pt"


def embedding_store(kind: str, fold: str) -> Path:
    """Per-movie embeddings (``<movie>.npy`` float16 (N, 768) and ``<movie>.parquet``) of every movie by the
    encoder of held-out group ``fold``; ``kind`` is ``adapted`` or ``finetuned``."""
    root = {"adapted": "dinov2", "finetuned": "dinov2_finetuned"}[kind]
    return paths.runs_dir() / root / "embeddings" / fold


# ------------------------------------------------------------------ encoder
def check_weights(path: Path | None = None) -> None:
    """Raise unless the checkpoint at ``path`` (default: torch.hub's cached copy) has the pinned sha256."""
    p = Path(path) if path is not None else Path(torch.hub.get_dir()) / "checkpoints" / f"{MODEL_NAME}_pretrain.pth"
    if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise RuntimeError(f"{p}: not the checkpoint of the paper (sha256 differs from {WEIGHTS_SHA256})")


def load_encoder(name: str, device: torch.device, repo_dir: Path | None = None):
    """The pretrained DINOv2 encoder from the pinned hub commit, or from a local copy of that commit
    (``repo_dir``), in eval mode on ``device``."""
    if repo_dir is not None:
        m = torch.hub.load(str(repo_dir), name, source="local", verbose=False)
    else:
        m = torch.hub.load(f"{HUB_REPO}:{HUB_COMMIT}", name, verbose=False, trust_repo=True, skip_validation=True)
    check_weights()
    return m.eval().to(device)


def load_adapted(model, weights, device, expect_backbone: str = "dinov2") -> dict:
    """Load an adapted or fine-tuned checkpoint into ``model``; any missing or unexpected key, or a checkpoint of
    another backbone family, is an error. Returns the checkpoint's recipe."""
    obj = torch.load(weights, map_location=device, weights_only=False)
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    recipe = obj.get("recipe", {}) if isinstance(obj, dict) else {}
    # the recipe names the model ("dinov2_vitb14"); the family is its prefix
    got = recipe.get("backbone")
    if got and not str(got).startswith(expect_backbone):
        raise SystemExit(f"FAIL: {Path(weights).name} was adapted from {got}, "
                         f"which is not the {expect_backbone} family")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"FAIL: {Path(weights).name} does not match the architecture "
                         f"({len(missing)} missing, {len(unexpected)} unexpected keys). "
                         f"Refusing to embed with a partially-loaded encoder.")
    model.eval()
    print(f"domain-adapted weights loaded from {weights} (recipe: {recipe or 'unrecorded'})",
          flush=True)
    return recipe


# ------------------------------------------------------------------ training pieces
def _blur_kernel(dev):
    """The 5 x 5 binomial blur kernel, (1, 1, 5, 5)."""
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=dev)
    k = (k[:, None] * k[None, :])
    return (k / k.sum()).view(1, 1, 5, 5)


def augment(x: torch.Tensor, g: torch.Generator, K: torch.Tensor) -> torch.Tensor:
    """One augmented view of a batch x (B, 1, H, W) in [0, 1], drawn per sample from ``g``.

    A resized crop covering 50-100 % of the field, flips and a 90-degree rotation in one affine warp, the blur
    ``K`` on half the samples, and an intensity scale of 1 +- 0.075 and shift of +- 0.03. The pixel value is phase,
    so there is no colour jitter and the intensity change is kept small.
    """
    B, _, H, W = x.shape
    dev = x.device
    r = lambda: torch.rand(B, generator=g, device=dev)
    s = 0.5 + 0.5 * r()                                       # crop side as a fraction of the field
    k = torch.randint(0, 4, (B,), generator=g, device=dev)    # 90-degree rotation, per sample
    fx = torch.where(r() < 0.5, -1.0, 1.0)                    # per-sample flips
    fy = torch.where(r() < 0.5, -1.0, 1.0)
    ang = k.float() * (math.pi / 2)
    cos, sin = torch.cos(ang), torch.sin(ang)
    lim = 1.0 - s                                             # keep the crop inside the frame
    theta = torch.stack([
        torch.stack([s * cos * fx, -s * sin * fy, (2 * r() - 1) * lim], 1),
        torch.stack([s * sin * fx,  s * cos * fy, (2 * r() - 1) * lim], 1)], 1)
    grid = Fn.affine_grid(theta, (B, 1, H, W), align_corners=False)
    out = Fn.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    m = (torch.rand(B, 1, 1, 1, generator=g, device=dev) < 0.5).to(out.dtype)
    out = m * Fn.conv2d(Fn.pad(out, (2, 2, 2, 2), mode="reflect"), K) + (1 - m) * out

    a = 1.0 + 0.15 * (torch.rand(B, 1, 1, 1, generator=g, device=dev) - 0.5)
    b = 0.06 * (torch.rand(B, 1, 1, 1, generator=g, device=dev) - 0.5)
    return (out * a + b).clamp(0, 1)


class Head(nn.Module):
    """Projection head of the contrastive objective: 768 -> 512 -> 128, L2-normalised."""

    def __init__(self, d=768, h=512, o=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, o))

    def forward(self, x):
        return Fn.normalize(self.net(x), dim=-1)


def nt_xent(z1, z2, t=0.2):
    """NT-Xent (SimCLR) loss of two batches of normalised projections, temperature ``t``."""
    z = torch.cat([z1, z2]); n = z1.shape[0]
    sim = (z @ z.T) / t
    sim.fill_diagonal_(-1e4)
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return Fn.cross_entropy(sim, tgt)


# ------------------------------------------------------------------ embedding
@torch.no_grad()
def embed_movie(arr: np.ndarray, model, device: torch.device, batch: int, res: int) -> np.ndarray:
    """Encoder output for every crop of ``arr`` (N, H, W) of raw phase. Returns (N, D) float16.

    Each crop goes through the global phase map, log1p(9 * clip((x - lo) / (hi - lo), 0, 1)) / ln 10, which is
    monotone in phase, is repeated to 3 channels, resized to ``res`` and ImageNet-normalised.
    """
    out = []
    for i in range(0, len(arr), batch):
        x = torch.from_numpy(np.asarray(arr[i : i + batch], dtype=np.float32)).to(device)
        lin = ((x - GLOBAL_LO) / (GLOBAL_HI - GLOBAL_LO)).clamp(0, 1)
        x = torch.log1p(9.0 * lin) / np.log(10.0)
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)                     # grayscale -> 3 channels
        x = torch.nn.functional.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
        x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
            f = model(x)
        # an encoder that returns an output object gives its pooled or class token
        if not torch.is_tensor(f):
            f = getattr(f, "pooler_output", None)
            if f is None:
                f = model(x).last_hidden_state[:, 0]
        out.append(f.float().cpu().numpy().astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, 768), dtype=np.float16)


def default_device(name: str | None = None) -> torch.device:
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))
