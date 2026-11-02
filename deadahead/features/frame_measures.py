"""Per-cell, per-frame shape and phase measures from the curated masks (``cells.parquet``).

    deadahead data-cells [--movie-table tables/movies.csv] [--raw-dir DIR] [--movies DU145_bp_1 ...]

Each movie's phase and mask stacks are read one frame at a time. Lengths use 0.629 um per pixel and times 3 min
per frame (the dataset's documentation; the files carry no metadata). Phase values are in pg/um^2, so the dry mass
of a cell is its summed phase times the pixel area, in pg. One shard per movie is written under
``tables/cells_shards/``; a movie whose shard exists is skipped, and the shards are joined into ``cells.parquet``
once every movie of the movie table has one.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table

from deadahead import paths
from deadahead.data.tables import MORPHOLOGY_EXCLUDED
from deadahead.survival.config import windows_config

PROPS = (
    "label",
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "extent",
    "centroid",
    "mean_intensity",
    "max_intensity",
    "min_intensity",
    "equivalent_diameter_area",
    "axis_major_length",
    "axis_minor_length",
    "orientation",
)


def extract_movie(image_path: str, mask_path: str, movie_id: str, px_um: float,
                  interval_min: float) -> pd.DataFrame:
    """Region measures of every masked cell in every frame of one movie, in pixels and in um.

    One row per (cell, frame) with cell_num, frame_index (0-based), track_id, timestamp_min and the measures.
    """
    out = []
    with tifffile.TiffFile(image_path) as fi, tifffile.TiffFile(mask_path) as fm:
        n = min(len(fi.pages), len(fm.pages))
        px_area = px_um * px_um
        for k in range(n):
            img = fi.pages[k].asarray()
            msk = fm.pages[k].asarray()
            if msk.dtype != np.int32:
                msk = msk.astype(np.int32)
            if not msk.any():
                continue
            t = regionprops_table(msk, intensity_image=img, properties=PROPS)
            df = pd.DataFrame(t)
            df["frame_index"] = k
            # phase is in pg/um^2, so the phase summed over the mask times the pixel area is the dry mass in pg
            df["dry_mass_pg"] = df["mean_intensity"] * df["area"] * px_area
            out.append(df)
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    d = d.rename(
        columns={
            "label": "cell_num",
            "area": "area_px",
            "perimeter": "perimeter_px",
            "centroid-0": "centroid_y_px",
            "centroid-1": "centroid_x_px",
            "mean_intensity": "mean_phase",
            "max_intensity": "max_phase",
            "min_intensity": "min_phase",
            "equivalent_diameter_area": "equiv_diameter_px",
            "axis_major_length": "axis_major_px",
            "axis_minor_length": "axis_minor_px",
        }
    )
    d["movie_id"] = movie_id
    d["timestamp_min"] = d["frame_index"] * interval_min
    d["area_um2"] = d["area_px"] * px_um * px_um
    d["perimeter_um"] = d["perimeter_px"] * px_um
    d["equiv_diameter_um"] = d["equiv_diameter_px"] * px_um
    d["axis_major_um"] = d["axis_major_px"] * px_um
    d["axis_minor_um"] = d["axis_minor_px"] * px_um
    d["centroid_x_um"] = d["centroid_x_px"] * px_um
    d["centroid_y_um"] = d["centroid_y_px"] * px_um
    d["aspect_ratio"] = d["axis_major_px"] / d["axis_minor_px"].replace(0, np.nan)
    # circularity 4*pi*A/P^2: 1 for a disc, lower for a ragged or fragmented outline
    d["circularity"] = 4 * np.pi * d["area_px"] / d["perimeter_px"].replace(0, np.nan) ** 2
    d["track_id"] = d["movie_id"] + ":" + d["cell_num"].astype(str)
    d["morphology_excluded"] = movie_id in MORPHOLOGY_EXCLUDED
    d["units_source"] = "documentation"  # pixel size and frame interval from the dataset's documentation
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write the per-cell, per-frame measures (cells.parquet).")
    ap.add_argument("--movie-table", type=Path, default=None, help="movie table (.csv); default: tables/movies.csv")
    ap.add_argument("--raw-dir", type=Path, default=None, help="extracted archives (default: raw/celldeathdetect)")
    ap.add_argument("--config", type=Path, default=None, help="windows settings (default: configs/windows.yaml)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: tables/")
    ap.add_argument("--movies", nargs="*", default=None, help="restrict to these movie ids")
    ap.add_argument("--force", action="store_true", help="recompute shards that exist")
    ap.add_argument("--no-concat", action="store_true", help="write the shards only")
    a = ap.parse_args(argv)
    table = a.movie_table or paths.tables_dir() / "movies.csv"
    raw = a.raw_dir or paths.raw_dir()
    out = a.out_dir or paths.tables_dir()
    if not table.is_file():
        print(f"movie table not found: {table} (run data-inventory first)", file=sys.stderr)
        return 3

    cfg = windows_config(a.config)
    px_um = float(cfg["pixel_size_um_per_px"])
    interval = float(cfg["frame_interval_min"])

    inv = pd.read_csv(table)
    all_movies = list(inv["movie_id"])
    if a.movies:
        unknown = sorted(set(a.movies) - set(all_movies))
        if unknown:
            print(f"not in the movie table: {unknown}", file=sys.stderr)
            return 2
        inv = inv[inv["movie_id"].isin(a.movies)]
    shards = out / "cells_shards"
    shards.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for i, r in enumerate(inv.itertuples(index=False), 1):
        shard = shards / f"{r.movie_id}.parquet"
        if shard.exists() and not a.force:
            continue
        t0 = time.time()
        d = extract_movie(str(raw / r.image_path), str(raw / r.mask_path), r.movie_id, px_um, interval)
        tmp = shard.with_suffix(".parquet.tmp")  # written whole, then renamed, so a partial shard is never read
        d.to_parquet(tmp, index=False)
        tmp.replace(shard)
        print(f"[{i}/{len(inv)}] {r.movie_id:<14} {len(d):>7} rows  {time.time() - t0:6.1f} s  "
              f"(elapsed {(time.time() - t_start) / 60:.1f} min)", flush=True)

    if a.no_concat:
        return 0
    present = [m for m in sorted(all_movies) if (shards / f"{m}.parquet").exists()]
    if len(present) < len(all_movies):
        print(f"{len(present)} of {len(all_movies)} movies have a shard; cells.parquet not written yet")
        return 0
    cells = pd.concat([pd.read_parquet(shards / f"{m}.parquet") for m in present], ignore_index=True)
    cells.to_parquet(out / "cells.parquet", index=False)
    print(f"{len(cells):,} rows, {cells['track_id'].nunique()} tracks, {len(present)} movies "
          f"-> {out / 'cells.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
