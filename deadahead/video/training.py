"""Shared parts of VideoMAE training: the phase map, augmentation, model input, tube masking, the
learning-rate schedule and resumable state files."""

from __future__ import annotations

import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as Fn

from .model import (FRAMES, IMAGE_SIZE, MASK_RATIO, PHASE_HI, PHASE_LO, SPATIAL_TOKENS, TEMPORAL_TOKENS,
                    InputError)

#: Effective batch for both training stages; ``micro_batch * grad_accum`` must equal it.
EFFECTIVE_BATCH = 128
#: The augmentation's torch generator is seeded at ``seed + AUG_STREAM_OFFSET``.
AUG_STREAM_OFFSET = 1


# ------------------------------------------------------------------ input pipeline
def phase_to_unit(x: torch.Tensor) -> torch.Tensor:
    """Raw phase to [0, 1]: ``lin = clip((x + 0.3) / 4.3, 0, 1)``, then ``log1p(9 * lin) / ln 10``."""
    lin = ((x.float() - PHASE_LO) / (PHASE_HI - PHASE_LO)).clamp(0, 1)
    return torch.log1p(9.0 * lin) / math.log(10.0)


def blur_kernel(dev) -> torch.Tensor:
    """5x5 binomial blur kernel."""
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=dev)
    k = k[:, None] * k[None, :]
    return (k / k.sum()).view(1, 1, 5, 5)


def augment_clip(x: torch.Tensor, g: torch.Generator, K: torch.Tensor) -> torch.Tensor:
    """Microscopy augmentation with one random draw per clip, applied identically to all T frames.

    ``x`` is ``(B, T, H, W)`` in [0, 1]: one affine warp per clip (random resized crop 50-100 %,
    flips, 90-degree rotation), a light blur on half the clips and a mild intensity scale and shift.
    No colour jitter: in phase imaging the pixel value is dry mass.
    """
    B, T, H, W = x.shape
    dev = x.device
    r = lambda: torch.rand(B, generator=g, device=dev)
    s = 0.5 + 0.5 * r()
    k = torch.randint(0, 4, (B,), generator=g, device=dev)
    fx = torch.where(r() < 0.5, -1.0, 1.0)
    fy = torch.where(r() < 0.5, -1.0, 1.0)
    ang = k.float() * (math.pi / 2)
    cos, sin = torch.cos(ang), torch.sin(ang)
    lim = 1.0 - s
    theta = torch.stack([
        torch.stack([s * cos * fx, -s * sin * fy, (2 * r() - 1) * lim], 1),
        torch.stack([s * sin * fx, s * cos * fy, (2 * r() - 1) * lim], 1)], 1)
    grid = Fn.affine_grid(theta, (B, 1, H, W), align_corners=False)
    out = Fn.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    m = (torch.rand(B, 1, 1, 1, generator=g, device=dev) < 0.5).to(out.dtype)
    blurred = Fn.conv2d(Fn.pad(out, (2, 2, 2, 2), mode="reflect"), K.repeat(T, 1, 1, 1), groups=T)
    out = m * blurred + (1 - m) * out
    a = 1.0 + 0.15 * (torch.rand(B, 1, 1, 1, generator=g, device=dev) - 0.5)
    b = 0.06 * (torch.rand(B, 1, 1, 1, generator=g, device=dev) - 0.5)
    return (out * a + b).clamp(0, 1)


def to_model_input(x: torch.Tensor, mean, std) -> torch.Tensor:
    """``(B, T, H, W)`` in [0, 1] to ``(B, T, 3, 224, 224)``: bilinear resize, three channels,
    ImageNet normalisation. Raises if the input is outside [0, 1]."""
    if not (float(x.min()) >= 0.0 and float(x.max()) <= 1.0):
        raise ValueError(f"model input outside [0, 1]: min {float(x.min()):.3f}, max {float(x.max()):.3f}")
    B, T = x.shape[:2]
    x = Fn.interpolate(x.reshape(B * T, 1, *x.shape[2:]), size=(IMAGE_SIZE, IMAGE_SIZE),
                       mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    m = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    s = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return ((x - m) / s).view(B, T, 3, IMAGE_SIZE, IMAGE_SIZE)


def clip_batch(gather, movies, rows, pick, dev, g_aug, K) -> torch.Tensor:
    """Gather, map, augment: one micro-batch of training clips, ``(B, T, H, W)`` in [0, 1]."""
    raw = torch.from_numpy(np.ascontiguousarray(gather.batch(movies[pick], [rows[i] for i in pick])))
    x = phase_to_unit(raw.to(dev, non_blocking=True))
    if x.shape[1] != FRAMES:
        raise ValueError(f"a clip has {x.shape[1]} frames, not {FRAMES}")
    return augment_clip(x, g_aug, K)


# ------------------------------------------------------------------ tube masking
def n_masked(ratio: float = MASK_RATIO, spatial: int = SPATIAL_TOKENS) -> int:
    """Number of the 196 spatial positions a tube mask hides (176 at 0.90)."""
    k = int(round(ratio * spatial))
    if not 0 < k < spatial:
        raise InputError(
            f"mask ratio {ratio} would hide {k} of {spatial} spatial positions; the objective "
            f"needs at least one visible and at least one masked position")
    return k


def tube_mask(batch: int, rng: np.random.Generator, *, ratio: float = MASK_RATIO,
              spatial: int = SPATIAL_TOKENS, temporal: int = TEMPORAL_TOKENS) -> np.ndarray:
    """Tube mask as ``(batch, temporal * spatial)`` bool, True where a token is hidden.

    Independent per clip; the same spatial positions are hidden in every temporal group.
    """
    k = n_masked(ratio, spatial)
    out = np.zeros((batch, temporal, spatial), dtype=bool)
    for i in range(batch):
        hide = rng.choice(spatial, size=k, replace=False)
        out[i, :, hide] = True
    return out.reshape(batch, temporal * spatial)


def is_tube(mask: np.ndarray, *, spatial: int = SPATIAL_TOKENS,
            temporal: int = TEMPORAL_TOKENS) -> bool:
    """True if every clip's mask is constant along time."""
    m = np.asarray(mask).reshape(-1, temporal, spatial)
    return bool((m == m[:, :1, :]).all())


# ------------------------------------------------------------------ schedule and state
class Split(NamedTuple):
    """How one effective batch is realised on one device: ``micro_batch * grad_accum`` rows.

    The split changes the order of random draws; repeating a run's sampling needs the same micro-batch.
    """

    micro_batch: int
    grad_accum: int

    @property
    def effective(self) -> int:
        return self.micro_batch * self.grad_accum


def split_batch(micro_batch: int, effective: int = EFFECTIVE_BATCH) -> Split:
    """Split the effective batch into ``effective // micro_batch`` accumulation steps."""
    if micro_batch <= 0 or effective % micro_batch:
        raise InputError(f"micro-batch {micro_batch} does not divide the effective batch {effective}")
    return Split(micro_batch, effective // micro_batch)


def warmup_cosine(opt, steps: int, warmup_fraction: float = 0.05):
    """Linear warm-up from 1e-3 of the rate over ``max(1, int(fraction * steps))`` steps, then cosine."""
    warm = max(1, int(warmup_fraction * steps))
    return torch.optim.lr_scheduler.SequentialLR(
        opt, [torch.optim.lr_scheduler.LinearLR(opt, 1e-3, 1.0, warm),
              torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, steps - warm))],
        milestones=[warm])


class Interrupt:
    """SIGTERM and, where the platform has it, SIGUSR1 set a flag; the training loop then saves its state and exits."""

    def __init__(self):
        self.hit = False
        for sig in (getattr(signal, name) for name in ("SIGTERM", "SIGUSR1") if hasattr(signal, name)):
            try:
                signal.signal(sig, self._set)
            except (ValueError, OSError):          # not the main thread: leave the default
                pass

    def _set(self, *_):
        self.hit = True


def atomic_save(obj, path: Path) -> None:
    """``torch.save`` to a temporary file, then rename, so a killed write never leaves a torn file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_state(path: Path, model, opt, sched, step_next: int, rng: np.random.Generator,
               g_aug: torch.Generator, losses: list, *, run: dict | None = None) -> None:
    """Everything a resumed run needs: weights, optimiser, schedule, step and all random streams.

    ``run`` is what identifies the run; ``load_state`` compares it.
    """
    atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "step": step_next,
                 "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                 "aug_rng": g_aug.get_state(), "losses": list(losses), "run": run}, path)


def load_state(path: Path, model, opt, sched, g_aug: torch.Generator, dev, *, run: dict | None = None):
    """Restore a state file written by ``save_state``. Returns ``(step, numpy rng, losses)``.

    With ``run`` given, raises InputError if the file belongs to another run: a schedule restored from a
    run of another length would corrupt the learning rate.
    """
    st = torch.load(path, map_location=dev, weights_only=False)
    if run is not None and st.get("run") != run:
        raise InputError(f"{path} belongs to another run ({st.get('run')}, not {run}); delete it to start again")
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    rng = np.random.default_rng(); rng.bit_generator.state = st["np_rng"]
    torch.set_rng_state(st["torch_rng"].cpu())
    g_aug.set_state(st["aug_rng"].cpu())      # generator states are CPU byte tensors
    return int(st["step"]), rng, list(st.get("losses", []))


def run_differs(ckpt: Path, run: dict, also: Path | None = None) -> bool:
    """True, with a message, if the ``.json`` record beside ``ckpt`` does not match ``run``. ``also`` is one more
    file of that run to delete with it."""
    record = ckpt.with_suffix(".json")
    done = json.loads(record.read_text()) if record.is_file() else {}
    differs = [f"{k} {done.get(k)}, not {v}" for k, v in run.items() if done.get(k) != v]
    if differs:
        files = "its .json and .state.pt files" + (f" and {also}" if also else "")
        print(f"{ckpt} was trained differently ({'; '.join(differs)}); delete it with {files} to train again",
              file=sys.stderr)
    return bool(differs)


def finished_run(ckpt: Path, run: dict) -> int:
    """Exit code of a command whose checkpoint exists: 0 if its ``.json`` record matches ``run``, else 3."""
    if run_differs(ckpt, run):
        return 3
    print(f"{ckpt} exists; nothing to do")
    return 0


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> float:
    return time.time()
