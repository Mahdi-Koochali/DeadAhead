"""Test-time changes to the fine-tuned video model: the tracked cell hidden, its 16 frames shuffled, or the last
frame repeated 16 times.

One trained model scores its held-out group under four conditions, through the prediction path of
``deadahead.video.predict`` with one step inserted between reading the raw clip and the phase map:

    intact               the clip as stored
    hidden_cell          the tracked cell's mask, dilated by 3 px, set to 0 (the crop store's pad value) in every frame
    shuffled_frames      the 16 frames in a fixed random order per row (never the identity)
    repeated_last_frame  the landmark frame repeated 16 times

The hidden-cell condition reads a mask store aligned row by row with the all-frames crop store:

    <masks>/<movie>.npy       uint8 (N, 128, 128); bit 0 the central cell's mask, bit 1 the same dilated 3 px,
                              bit 2 any annotated cell
    <masks>/<movie>.fill.npy  float16 (N,): the background phase of each row's frame
    <masks>/<movie>.json      the movie's counts and checks (written last: marks the movie done)

``masks`` re-reads every movie's record and fails while any check in ``MASK_CHECKS`` has failed, on a re-run too.

    deadahead video-test-time masks [--movies M ...] [--movie-table CSV] [--cells FILE] [--out-dir DIR]
    deadahead video-test-time predict --fold G --seed S (--weights FILE | --backbone-lr LR --layer-decay D)
                              [--masks DIR] [--store DIR] [--windows FILE] [--out-dir DIR] [--device DEV]

``predict`` writes <out-dir>/<condition>/<group>__s<seed>.parquet with the columns of ``deadahead.video.predict``.
GPU code: one fold and seed take a few minutes on one recent data-centre GPU (four conditions, bf16).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from deadahead import paths

#: the conditions, in the order they are scored
CONDITIONS = ("intact", "hidden_cell", "shuffled_frames", "repeated_last_frame")
ALTERED = CONDITIONS[1:]
#: the crop store's pad value, written over the hidden pixels
FILL_ZERO = 0.0
#: the background level: median raw phase over pixels farther than this from every annotated mask
BACKGROUND_MARGIN_PX = 12
#: the hidden region is the central cell's mask dilated by a disc of this radius
DILATION_PX = 3
#: the mask store's bit planes
BIT_OWN, BIT_OWN_DIL, BIT_ANY = 1, 2, 4
#: which plane each mask condition hides, and which fill it writes there
MASK_RULE = {"hidden_cell": (BIT_OWN_DIL, "zero")}
#: the permutation seed, the clip length and the crop size
PERM_SEED = 2601562
FRAMES = 16
CROP = 128
#: the fold index k of the permutation streams: a group's position in this order
FOLD_ORDER = ("PNT1A_bp", "PNT1A_do", "PNT1A_st", "DU145_bp", "DU145_do", "DU145_st", "LNCaP_bp", "LNCaP_do",
              "LNCaP_st")
#: sha256 of each fold's permutations (int64, row order) under numpy 1.26 and 2.x; a generator whose streams
#: changed would give other frame orders, and ``predict`` refuses them
PERM_SHA256 = {
    "PNT1A_bp": "ab40a12ea4b0d177b3a969c97260ff8d15feb76423821ce918ededc1411e1354",
    "PNT1A_do": "f5788cda9aabd2e4de7ae8db6fe527bfbc2881291a1c39bf6e215bcff346a07d",
    "PNT1A_st": "92c3f1d59f5c9e40ca2e4aa14a158b3588cc154e13e824976584169d53ed4303",
    "DU145_bp": "18bc258c7521faa1ed883ebc5270c520ea86d16fb9f2309ca49bea9f7a693af3",
    "DU145_do": "44155ffa6457598eba6195bab32935b4c19eb3be1f7146dd16bbd35a7d7bbc33",
    "DU145_st": "fff2e2d564eb1c6d79d9131b9f5e205788e316328135e4e5bbeed9e20387a167",
    "LNCaP_bp": "fc85e686c2ac18a88ca0cecaa68f49925c37143944c5d73d612980f7d8e582a8",
    "LNCaP_do": "2a69a3eb7c8bf6f1fd58d6fb1da0cd59917cf2a84a5a49b3f12d7151d7732b3e",
    "LNCaP_st": "ae836876bf849a18aa0c54ed493fb07132970432981a85a1170c53f0ae9e89c4",
}
#: the mean phase of the image crop over the mask against the cell table (float16 storage of the crops)
MEAN_TOLERANCE = 0.001
#: the fields of a movie's record that must all be zero for the store to be used
MASK_CHECKS = ("area_check_failures", "mean_check_failures", "frames_with_non_integer_labels",
               "frames_with_no_background_pixel", "rows_with_empty_central_mask_in_crop")
PRED_BATCH = 64


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def disc(radius: int = DILATION_PX) -> np.ndarray:
    """Every offset with dy^2 + dx^2 <= radius^2, as a (2r+1, 2r+1) boolean structuring element."""
    r = int(radius)
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (y * y + x * x) <= r * r


def dilate(mask: np.ndarray, radius: int = DILATION_PX) -> np.ndarray:
    """Binary dilation of a 2-D boolean mask by ``disc(radius)``; pixels outside the array are empty."""
    from scipy.ndimage import binary_dilation
    return binary_dilation(np.asarray(mask, dtype=bool), structure=disc(radius))


def window(arr: np.ndarray, cy: float, cx: float, size: int = CROP, margin: int = 0, pad=0) -> np.ndarray:
    """The crop store's window (``deadahead.video.crops.crop_one``: ``int(round(c)) - size // 2``), widened by
    ``margin`` on every side; out-of-field pixels are ``pad``. With margin 0 it is the image crop's window."""
    h, w = arr.shape
    half = size // 2
    y0, x0 = int(round(cy)) - half - margin, int(round(cx)) - half - margin
    side = size + 2 * margin
    y1, x1 = y0 + side, x0 + side
    out = np.full((side, side), pad, dtype=arr.dtype)
    sy0, sx0, sy1, sx1 = max(y0, 0), max(x0, 0), min(y1, h), min(x1, w)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def mask_planes(label_frame: np.ndarray, cell_num: int, cy: float, cx: float, *, size: int = CROP,
                radius: int = DILATION_PX) -> np.ndarray:
    """One row of the mask store: uint8 ``(size, size)`` with three bit planes: the central cell's mask, the same
    dilated by ``radius`` px, and any annotated cell. ``label_frame`` is the label image of the frame (0 = no cell).
    The dilation is computed on a window widened by ``radius`` and cut back, then limited to pixels inside the field,
    so it equals the crop of the full-frame dilation and no pad pixel is in any plane."""
    m = int(radius)
    wide = window(label_frame, cy, cx, size, m, 0)
    h, w = label_frame.shape
    ys = np.arange(size) + (int(round(cy)) - size // 2)
    xs = np.arange(size) + (int(round(cx)) - size // 2)
    in_field = ((ys >= 0) & (ys < h))[:, None] & ((xs >= 0) & (xs < w))[None, :]
    own_dil = dilate(wide == int(cell_num), m)[m:m + size, m:m + size] & in_field
    inner = wide[m:m + size, m:m + size]
    out = np.zeros((size, size), dtype=np.uint8)
    out |= (inner == int(cell_num)).astype(np.uint8) * BIT_OWN
    out |= own_dil.astype(np.uint8) * BIT_OWN_DIL
    out |= (inner > 0).astype(np.uint8) * BIT_ANY
    return out


def background_level(image_frame: np.ndarray, label_frame: np.ndarray, margin: int = BACKGROUND_MARGIN_PX) -> float:
    """The frame's background phase: the median raw phase over pixels farther than ``margin`` px from every
    annotated mask (Euclidean); nan when the frame has no such pixel."""
    from scipy.ndimage import distance_transform_edt
    far = distance_transform_edt(np.asarray(label_frame) == 0) > margin
    if not far.any():
        return float("nan")
    return float(np.median(np.asarray(image_frame)[far]))


def build_movie(movie: str, group: str, image_path: str, mask_path: str, out_dir: str, crops_dir: str,
                cells_path: str) -> dict:
    """One movie's mask store, aligned row by row with its crop store shard, and each row's background level.

    Two checks tie the masks to the crops and the cell table: the full-frame pixel count of the central label equals
    ``area_px`` for every row, and, for rows not clipped at the border whose mask lies inside the crop, the mean of
    the image crop over the mask equals ``mean_phase`` within ``MEAN_TOLERANCE``. Returns the movie's counts."""
    import tifffile
    out_dir, crops_dir = Path(out_dir), Path(crops_dir)
    t0 = time.time()
    idx = pd.read_parquet(crops_dir / f"{movie}.parquet").sort_values("row").reset_index(drop=True)
    if not np.array_equal(idx["row"].to_numpy(), np.arange(len(idx))):
        raise RuntimeError(f"{movie}: the crop index rows are not 0..N-1")
    cells = pd.read_parquet(cells_path, columns=["movie_id", "cell_num", "frame_index", "centroid_y_px",
                                                 "centroid_x_px", "area_px", "mean_phase"],
                            filters=[("movie_id", "==", movie)])
    m = idx.merge(cells, on=["cell_num", "frame_index"], how="left", validate="one_to_one")
    if m["area_px"].isna().any() or len(m) != len(idx):
        raise RuntimeError(f"{movie}: {int(m['area_px'].isna().sum())} crop rows have no cells.parquet row")
    img = np.load(crops_dir / f"{movie}.npy", mmap_mode="r")
    if img.shape != (len(idx), CROP, CROP):
        raise RuntimeError(f"{movie}: image shard shape {img.shape} does not match {len(idx)} index rows")

    tmp = out_dir / f"{movie}.npy.tmp"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(idx), CROP, CROP))
    fill = np.full(len(idx), np.nan, dtype=np.float64)
    area_bad, mean_bad, mean_checked, max_mean_diff, own_empty = 0, 0, 0, 0.0, 0
    noninteger, no_background = 0, 0
    bg_by_frame = {}
    sums = np.zeros(3, dtype=np.float64)
    with tifffile.TiffFile(mask_path) as fm, tifffile.TiffFile(image_path) as fi:
        n_pages = min(len(fm.pages), len(fi.pages))
        for frame, grp in m.groupby("frame_index", sort=True):
            if int(frame) >= n_pages:
                raise RuntimeError(f"{movie}: crop rows at frame {frame}, but the stacks have {n_pages} pages")
            raw = fm.pages[int(frame)].asarray()
            lab = raw.astype(np.int32)
            if not np.array_equal(lab, raw):
                noninteger += 1
            counts = np.bincount(lab.ravel())
            bg = background_level(fi.pages[int(frame)].asarray(), lab)
            no_background += not np.isfinite(bg)
            bg_by_frame[int(frame)] = bg
            for rec in grp.itertuples(index=False):
                c = int(rec.cell_num)
                planes = mask_planes(lab, c, rec.centroid_y_px, rec.centroid_x_px)
                arr[rec.row] = planes
                fill[rec.row] = bg
                full = int(counts[c]) if c < len(counts) else 0
                if full != int(round(rec.area_px)) or full == 0:
                    area_bad += 1
                own = (planes & BIT_OWN) != 0
                n_own = int(own.sum())
                own_empty += n_own == 0
                if not rec.border_clipped and n_own == full and full > 0:
                    diff = abs(float(np.asarray(img[rec.row], dtype=np.float64)[own].mean()) - float(rec.mean_phase))
                    mean_checked += 1
                    max_mean_diff = max(max_mean_diff, diff)
                    mean_bad += diff > MEAN_TOLERANCE
                for k, bit in enumerate((BIT_OWN, BIT_OWN_DIL, BIT_ANY)):
                    sums[k] += int(np.count_nonzero(planes & bit))
    arr.flush()
    del arr
    final = out_dir / f"{movie}.npy"
    tmp.replace(final)
    fill_p = out_dir / f"{movie}.fill.npy"
    ftmp = out_dir / f"{movie}.fill.tmp.npy"
    np.save(ftmp, fill.astype(np.float16))
    ftmp.replace(fill_p)
    px = float(len(idx) * CROP * CROP)
    bgv = np.array([v for v in bg_by_frame.values() if np.isfinite(v)], dtype=float)
    rec = {"movie_id": movie, "group": group, "rows": int(len(idx)), "sha256": sha256_file(final),
           "fill_sha256": sha256_file(fill_p),
           "area_check_failures": int(area_bad), "mean_check_rows": int(mean_checked),
           "mean_check_failures": int(mean_bad), "mean_check_max_abs_diff": float(max_mean_diff),
           "rows_with_empty_central_mask_in_crop": int(own_empty),
           "frames_with_non_integer_labels": int(noninteger), "frames_with_no_background_pixel": int(no_background),
           "frames": int(len(bg_by_frame)),
           "background_level": {"median_over_frames": float(np.median(bgv)) if len(bgv) else None,
                                "min": float(bgv.min()) if len(bgv) else None,
                                "max": float(bgv.max()) if len(bgv) else None},
           "pixel_fraction": {"own": sums[0] / px, "own_dilated": sums[1] / px, "any": sums[2] / px},
           "seconds": round(time.time() - t0, 1)}
    (out_dir / f"{movie}.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def permutation(fold_index: int, row_index: int) -> np.ndarray:
    """The frame order of row ``row_index`` of fold ``fold_index`` (its position in ``FOLD_ORDER``):
    ``default_rng([2601562, k, i]).permutation(16)``; an identity draw takes the generator's next draw."""
    rng = np.random.default_rng([PERM_SEED, int(fold_index), int(row_index)])
    p = rng.permutation(FRAMES)
    while np.array_equal(p, np.arange(FRAMES)):
        p = rng.permutation(FRAMES)
    return p.astype(np.int64)


def permutations(fold_index: int, n_rows: int) -> np.ndarray:
    """``(n_rows, 16)``: the frame order of every row of a fold, the same for the fold's three seeds."""
    return np.stack([permutation(fold_index, i) for i in range(int(n_rows))])


def permutations_sha256(perms: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(perms, dtype=np.int64).tobytes()).hexdigest()


def apply_condition(condition: str, clips: np.ndarray, planes: np.ndarray | None = None,
                    perms: np.ndarray | None = None, background: np.ndarray | None = None) -> np.ndarray:
    """One condition's transform of a batch of raw clips (numpy), as a new array.

    ``clips`` is ``(B, 16, H, W)`` raw phase, oldest frame first, index 15 the landmark frame; ``planes`` the matching
    uint8 mask-store rows; ``perms`` ``(B, 16)``; ``background`` ``(B, 16)``, each frame's background level. The
    dtype of ``clips`` is kept."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; known: {CONDITIONS}")
    clips = np.asarray(clips)
    if clips.ndim != 4 or clips.shape[1] != FRAMES:
        raise ValueError(f"expected clips of shape (B, {FRAMES}, H, W), got {clips.shape}")
    if condition == "intact":
        return clips.copy()
    if condition in MASK_RULE:
        if planes is None or np.asarray(planes).shape != clips.shape:
            raise ValueError(f"{condition} needs mask planes of the clips' shape {clips.shape}")
        bit, fill = MASK_RULE[condition]
        hide = (np.asarray(planes) & np.uint8(bit)) != 0
        if fill == "zero":
            value = np.asarray(FILL_ZERO, dtype=clips.dtype)
        else:
            if background is None or np.asarray(background).shape != clips.shape[:2]:
                raise ValueError(f"{condition} needs one background level per frame, shape {clips.shape[:2]}")
            if not np.all(np.isfinite(np.asarray(background, dtype=float))):
                raise ValueError(f"{condition}: a background level is not finite")
            value = np.asarray(background, dtype=clips.dtype)[:, :, None, None]
        return np.where(hide, value, clips).astype(clips.dtype, copy=False)
    if condition == "shuffled_frames":
        if perms is None or np.asarray(perms).shape != (clips.shape[0], FRAMES):
            raise ValueError(f"shuffled_frames needs one permutation of {FRAMES} per clip")
        perms = np.asarray(perms)
        if not np.array_equal(np.sort(perms, axis=1), np.tile(np.arange(FRAMES), (len(perms), 1))):
            raise ValueError("shuffled_frames: a row of perms is not a permutation of the 16 frame indices")
        return clips[np.arange(len(clips))[:, None], perms]
    return np.repeat(clips[:, -1:, :, :], FRAMES, axis=1)


class FillGatherer:
    """Each row's background level, read from the mask store's <movie>.fill.npy."""

    def __init__(self, mask_store: Path):
        self.dir = Path(mask_store)
        self._a: dict[str, np.ndarray] = {}

    def batch(self, movies, frame_rows_list) -> np.ndarray:
        out = []
        for m, r in zip(movies, frame_rows_list):
            a = self._a.get(m)
            if a is None:
                a = self._a[m] = np.load(self.dir / f"{m}.fill.npy")
            out.append(a[np.asarray(r, dtype=np.int64)])
        return np.stack(out)


def apply_condition_torch(condition: str, raw, planes, perms, background):
    """``apply_condition`` on tensors (the same transform, on the device). ``raw`` is ``(B, 16, H, W)`` in the
    store's dtype, ``planes`` the matching uint8 mask rows, ``perms`` a ``(B, 16)`` long tensor, ``background``
    ``(B, 16)``. Returns (the transformed clip, the share of pixels overwritten or None)."""
    import torch
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; known: {CONDITIONS}")
    if condition == "intact":
        return raw, None
    if condition in MASK_RULE:
        bit, fill = MASK_RULE[condition]
        hide = (planes & bit) != 0
        share = float(hide.float().mean())
        if fill == "zero":
            return raw.masked_fill(hide, FILL_ZERO), share
        if not bool(torch.isfinite(background.float()).all()):
            raise ValueError(f"{condition}: a background level is not finite")
        return torch.where(hide, background.to(raw.dtype)[:, :, None, None], raw), share
    if condition == "shuffled_frames":
        return raw[torch.arange(raw.shape[0], device=raw.device)[:, None], perms], None
    return raw[:, -1:].expand(-1, FRAMES, -1, -1), None


def forward_conditions(model, rows, store: Path, mask_store: Path, mean, std, dev, perms: np.ndarray, *,
                       conditions=CONDITIONS, batch: int = PRED_BATCH, deadline: float | None = None):
    """``{condition: (N, 3) softmax probabilities}`` for the rows of ``rows``, in row order, and the share of pixels
    each mask condition overwrote. The prediction path of ``deadahead.video.predict.predict_probs`` with one step
    inserted: the raw clip is transformed before the phase map. Raises LookupError when a frame of a clip has an
    empty central mask and TimeoutError after ``deadline``."""
    import torch
    from deadahead.video import training as T
    from deadahead.video.clips import ClipGatherer
    from deadahead.video.model import FRAMES

    gather, mgather, fgather = ClipGatherer(store), ClipGatherer(mask_store, dtype=np.uint8), FillGatherer(mask_store)
    need_masks = any(c in MASK_RULE for c in conditions)
    movies = rows["movie_id"].to_numpy()
    frows = rows["frame_rows"].to_numpy()
    n = len(rows)
    out = {c: np.empty((n, 3), dtype=np.float64) for c in conditions}
    filled = {c: 0.0 for c in conditions if c in MASK_RULE}
    perms_t = torch.from_numpy(np.ascontiguousarray(perms)).long().to(dev)
    try:
        with torch.no_grad():
            for s in range(0, n, batch):
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(f"the pass reached its time limit after {s} of {n} rows")
                pick = np.arange(s, min(s + batch, n))
                rr = [frows[i] for i in pick]
                raw = torch.from_numpy(np.ascontiguousarray(gather.batch(movies[pick], rr))).to(dev, non_blocking=True)
                if raw.shape[1] != FRAMES:
                    raise ValueError(f"a clip has {raw.shape[1]} frames, not {FRAMES}")
                planes = bgl = None
                if need_masks:
                    planes = torch.from_numpy(np.ascontiguousarray(mgather.batch(movies[pick], rr))).to(dev, non_blocking=True)
                    bgl = torch.from_numpy(np.ascontiguousarray(fgather.batch(movies[pick], rr))).to(dev, non_blocking=True)
                    if planes.shape != raw.shape:
                        raise ValueError(f"clip {tuple(raw.shape)} / mask {tuple(planes.shape)}: not one shape")
                    if not bool(((planes & BIT_OWN) != 0).flatten(2).any(dim=2).all()):
                        raise LookupError(f"a frame of a clip in rows {s}..{s + len(pick) - 1} has an empty central mask")
                for c in conditions:
                    xc, share = apply_condition_torch(c, raw, planes, perms_t[pick[0]:pick[-1] + 1], bgl)
                    if c in filled:
                        filled[c] += share * len(pick)
                    x = T.phase_to_unit(xc)
                    inp = T.to_model_input(x, mean, std)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        logits = model(pixel_values=inp).logits
                    out[c][pick] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    finally:
        gather.close()
        mgather.close()
    return out, {c: v / n for c, v in filled.items()}


# ------------------------------------------------------------------ the command
def default_masks() -> Path:
    return paths.crops_dir() / "masks128_all"


def default_out_dir() -> Path:
    return paths.runs_dir() / "videomae" / "test_time"


def masks_main(a) -> int:
    from deadahead.video.clips import default_store, input_missing, unknown_movies
    crops = a.store or default_store()
    out_dir = a.out_dir or default_masks()
    movie_table = a.movie_table or paths.tables_dir() / "movies.csv"
    cells = a.cells or paths.tables_dir() / "cells.parquet"
    if input_missing(movie_table, "data-inventory") or input_missing(cells, "data-cells"):
        return 3
    inv = pd.read_csv(movie_table)
    if a.movies:
        if unknown_movies(a.movies, inv["movie_id"]):
            return 2
        inv = inv[inv["movie_id"].isin(a.movies)]
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(inv.itertuples(index=False), 1):
        if all((out_dir / f"{r.movie_id}{s}").exists() for s in (".npy", ".fill.npy", ".json")):
            continue
        image, mask = (Path(p) if Path(p).is_absolute() else paths.raw_dir() / p for p in (r.image_path, r.mask_path))
        group = f"{r.cell_line}_{r.treatment}"
        rec = build_movie(r.movie_id, group, str(image), str(mask), str(out_dir), str(crops), str(cells))
        print(f"[{i}/{len(inv)}] {r.movie_id:<14} {rec['rows']:>6} rows, check failures "
              f"{sum(rec[k] for k in MASK_CHECKS)}, {rec['seconds']:.1f} s", flush=True)
    recs = [json.loads((out_dir / f"{r.movie_id}.json").read_text()) for r in inv.itertuples(index=False)]
    bad = {k: sum(rec[k] for rec in recs) for k in MASK_CHECKS}
    if any(bad.values()):
        movies = [rec["movie_id"] for rec in recs if any(rec[k] for k in MASK_CHECKS)]
        print(f"error: the mask store failed its checks: {bad}; movies {movies}. Delete their files under {out_dir} "
              f"to build them again", file=sys.stderr)
        return 1
    return 0


def predict_main(a) -> int:
    from deadahead.video.clips import default_store, default_windows, input_missing
    from deadahead.video.finetune import CLASSES, checkpoint_path
    from deadahead.video.finetune import default_out_dir as default_finetune_dir
    from deadahead.video.model import (check_weights_name, image_stats, load_classifier, load_released_classifier,
                                       pretrained_snapshot, resolve_device)
    from deadahead.video.predict import held_out_rows, prediction_frame

    if a.weights is not None:
        check_weights_name(a.weights, a.fold, a.seed)
    store = a.store or default_store()
    masks = a.masks or default_masks()
    windows = a.windows or default_windows()
    if input_missing(windows, "data-windows"):
        return 3
    rows = held_out_rows(windows, store, a.fold)
    perms = permutations(FOLD_ORDER.index(a.fold), len(rows))
    if permutations_sha256(perms) != PERM_SHA256[a.fold]:
        raise SystemExit("error: this numpy draws other frame orders than the released predictions")
    snap = a.snapshot or pretrained_snapshot()
    mean, std = image_stats(snap)
    dev = resolve_device(a.device)
    if a.weights is not None:
        model = load_released_classifier(a.weights, snap, dev)
    else:
        ckpt = checkpoint_path(a.runs_dir or default_finetune_dir(), a.fold, a.backbone_lr, a.layer_decay,
                               a.seed, final=True)
        model = load_classifier(ckpt, snap, dev)
    t0 = time.time()
    probs, filled = forward_conditions(model, rows, store, masks, mean, std, dev, perms, batch=a.batch)
    bad = [c for c, p in probs.items() if not np.isfinite(p).all()]
    if bad:
        raise SystemExit(f"error: non-finite probabilities under {bad}")
    out_dir = a.out_dir or default_out_dir()
    for c in CONDITIONS:
        d = out_dir / c
        d.mkdir(parents=True, exist_ok=True)
        prediction_frame(rows, probs[c], CLASSES).to_parquet(d / f"{a.fold}__s{a.seed}.parquet", index=False)
    print(f"{a.fold} s{a.seed}: {len(rows)} rows x {len(CONDITIONS)} conditions in {time.time() - t0:.0f} s -> "
          f"{out_dir}; share of pixels hidden {json.dumps(filled)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="deadahead video-test-time", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    m = sub.add_parser("masks", help="build the mask store (CPU)")
    m.add_argument("--movies", nargs="*", default=None, help="movie ids (default: all)")
    m.add_argument("--movie-table", type=Path, default=None,
                   help="CSV with movie_id, cell_line, treatment, image_path and mask_path (default: tables/movies.csv)")
    m.add_argument("--cells", type=Path, default=None, help="per-cell table (default: tables/cells.parquet)")
    m.add_argument("--store", type=Path, default=None, help="the all-frames crop store")
    m.add_argument("--out-dir", type=Path, default=None)
    p = sub.add_parser("predict", help="score one fine-tuned model under the four conditions (GPU)")
    p.add_argument("--fold", required=True, choices=FOLD_ORDER, help="held-out group")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--weights", type=Path, default=None, help="a released .safetensors file")
    p.add_argument("--backbone-lr", type=float, default=None, help="setting of the final run (with --layer-decay)")
    p.add_argument("--layer-decay", type=float, default=None)
    p.add_argument("--runs-dir", type=Path, default=None, help="fine-tuning checkpoints")
    p.add_argument("--snapshot", type=Path, default=None)
    p.add_argument("--masks", type=Path, default=None, help="the mask store")
    p.add_argument("--store", type=Path, default=None)
    p.add_argument("--windows", type=Path, default=None)
    p.add_argument("--batch", type=int, default=PRED_BATCH)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    if a.command == "masks":
        return masks_main(a)
    if a.weights is None and (a.backbone_lr is None or a.layer_decay is None):
        p.error("give --weights, or --backbone-lr and --layer-decay of the final run")
    return predict_main(a)


if __name__ == "__main__":
    raise SystemExit(main())
