"""All-frames crop store for the video model: one 128 x 128 crop per cell and frame, centred on the
cell's centroid, raw phase in float16, zero-padded where the crop runs off the field of view.

Per movie, under ``crops/frames128_all/``: ``<movie_id>.npy`` (N, 128, 128) and ``<movie_id>.parquet``
(movie_id, cell_num, track_id, frame_index, row, border_clipped). No normalisation is applied here.
Resumable: a movie whose two files exist is skipped.

    deadahead video-crops [--movies DU145_bp_1 ...]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from deadahead import paths

from .clips import default_store, input_missing, unknown_movies

SIZE = 128             # px; 0.629 um per px
STRIDE = 1             # every frame
PAD = 0.0              # background phase is about 0


def crop_one(img: np.ndarray, cy: float, cx: float, size: int, pad: float) -> tuple[np.ndarray, bool]:
    """Fixed-size crop centred on a centroid, padded with ``pad`` where it runs off the field."""
    h, w = img.shape
    half = size // 2
    y0, x0 = int(round(cy)) - half, int(round(cx)) - half
    y1, x1 = y0 + size, x0 + size
    clipped = y0 < 0 or x0 < 0 or y1 > h or x1 > w
    if not clipped:
        return img[y0:y1, x0:x1], False
    out = np.full((size, size), pad, dtype=img.dtype)
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y1, h), min(x1, w)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out, True


def extract_movie(image_path: str, movie_id: str, cells: pd.DataFrame,
                  size: int, stride: int, pad: float) -> tuple[np.ndarray, pd.DataFrame]:
    """Crops of one movie, ordered by frame then cell number, and their index table."""
    sub = cells[(cells["movie_id"] == movie_id) & (cells["frame_index"] % stride == 0)]
    sub = sub.sort_values(["frame_index", "cell_num"])
    if sub.empty:
        return np.zeros((0, size, size), dtype=np.float16), pd.DataFrame()

    out = np.zeros((len(sub), size, size), dtype=np.float16)
    rows, r = [], 0
    with tifffile.TiffFile(image_path) as fi:
        n_pages = len(fi.pages)
        for frame, grp in sub.groupby("frame_index", sort=True):
            if frame >= n_pages:
                continue
            img = fi.pages[int(frame)].asarray()
            for rec in grp.itertuples(index=False):
                c, clipped = crop_one(img, rec.centroid_y_px, rec.centroid_x_px, size, pad)
                out[r] = c.astype(np.float16)
                rows.append(
                    {
                        "movie_id": movie_id,
                        "cell_num": int(rec.cell_num),
                        "track_id": rec.track_id,
                        "frame_index": int(frame),
                        "row": r,
                        "border_clipped": bool(clipped),
                    }
                )
                r += 1
    return out[:r], pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the all-frames 128-px crop store of the video model.")
    ap.add_argument("--movies", nargs="*", default=None, help="movie ids (default: all)")
    ap.add_argument("--movie-table", type=Path, default=None,
                    help="CSV with movie_id and image_path (default: tables/movies.csv); "
                         "relative image paths are read under the raw data folder")
    ap.add_argument("--cells", type=Path, default=None, help="per-cell table (default: tables/cells.parquet)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.stride != STRIDE and a.out_dir is None:
        ap.error("--stride other than 1 needs --out-dir")
    out_dir = a.out_dir or default_store()
    movie_table = a.movie_table or paths.tables_dir() / "movies.csv"
    cells_file = a.cells or paths.tables_dir() / "cells.parquet"
    if input_missing(movie_table, "data-inventory") or input_missing(cells_file, "data-cells"):
        return 3
    inv = pd.read_csv(movie_table)
    if a.movies:
        if unknown_movies(a.movies, inv["movie_id"]):
            return 2
        inv = inv[inv["movie_id"].isin(a.movies)]
    cells = pd.read_parquet(
        cells_file,
        columns=["movie_id", "cell_num", "track_id", "frame_index", "centroid_y_px", "centroid_x_px"],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, r in enumerate(inv.itertuples(index=False), 1):
        npy, idx = out_dir / f"{r.movie_id}.npy", out_dir / f"{r.movie_id}.parquet"
        if npy.exists() and idx.exists() and not a.force:
            continue
        image = Path(r.image_path)
        image = image if image.is_absolute() else paths.raw_dir() / image
        arr, df = extract_movie(str(image), r.movie_id, cells, a.size, a.stride, PAD)
        # the index is written last, so a half-written pair never looks complete
        np.save(npy, arr)
        df.to_parquet(idx, index=False)
        print(f"[{i}/{len(inv)}] {r.movie_id:<14} {arr.shape[0]:>7} crops  "
              f"(elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
