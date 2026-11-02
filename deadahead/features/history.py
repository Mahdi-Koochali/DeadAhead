"""Handcrafted history features: 14 per-cell QPI measures summarised over each cell's own past.

Each of the 14 measures in ``BASE_FEATURES`` is summarised by 8 statistics over the frames t - W to t (W = 40 frames =
120 min), giving 14 x 8 = 112 features per cell and frame. A track has one row per frame, so this window is W + 1 rows.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from deadahead.paths import runs_dir, tables_dir

# Instantaneous shape and phase measures of one cell in one frame. No absolute time, no movie or group identity.
BASE_FEATURES = [
    "area_um2", "perimeter_um", "eccentricity", "solidity", "extent",
    "circularity", "aspect_ratio", "equiv_diameter_um",
    "axis_major_um", "axis_minor_um",
    "mean_phase", "max_phase", "min_phase", "dry_mass_pg",
]

# The statistics computed for every measure, in column order.
STATISTICS = ["last", "mean", "std", "min", "max", "delta", "delta10", "relchange"]

# The 112 feature columns in the order build_history_features writes them.
HISTORY_COLUMNS = [f"{f}__{s}" for f in BASE_FEATURES for s in STATISTICS]

HISTORY_FRAMES_W = 40


def build_history_features(cells: pd.DataFrame, W: int) -> pd.DataFrame:
    """Rolling summaries of each cell's past over a window of W + 1 frames ending at each frame.

    ``cells`` has one row per (track_id, frame_index) with the columns in ``BASE_FEATURES``. A row at frame t only
    reads frames t - W to t of the same track. Returns track_id, frame_index and the 112 columns of
    ``HISTORY_COLUMNS``; the windowed statistics are NaN until a track has W + 1 frames.
    """
    cells = cells.sort_values(["track_id", "frame_index"])
    g = cells.groupby("track_id", sort=False)
    out = {"track_id": cells["track_id"].to_numpy(), "frame_index": cells["frame_index"].to_numpy()}
    win = W + 1  # inclusive of both endpoints: [t-W, t]
    for f in BASE_FEATURES:
        s = cells[f]
        r = s.groupby(cells["track_id"], sort=False).rolling(win, min_periods=win)
        out[f"{f}__last"] = s.to_numpy()
        out[f"{f}__mean"] = r.mean().to_numpy()
        out[f"{f}__std"] = r.std().to_numpy()
        out[f"{f}__min"] = r.min().to_numpy()
        out[f"{f}__max"] = r.max().to_numpy()
        # net change over the window and over the last 10 frames (30 min): direction of travel, not level
        out[f"{f}__delta"] = (s - g[f].shift(W)).to_numpy()
        out[f"{f}__delta10"] = (s - g[f].shift(10)).to_numpy()
        # relative change, which is comparable across groups whose measures differ in scale
        prev = g[f].shift(W)
        out[f"{f}__relchange"] = ((s - prev) / prev.replace(0, np.nan)).to_numpy()
    df = pd.DataFrame(out)
    return df


def history_columns(hist: pd.DataFrame) -> list[str]:
    """The feature columns of a history table, in table order (every column but the two keys)."""
    return [c for c in hist.columns if c not in ("track_id", "frame_index")]


def history_features(W: int = HISTORY_FRAMES_W, cells_path: Path | None = None, cache: bool = True) -> pd.DataFrame:
    """``build_history_features`` for every cell of ``cells.parquet``, cached under ``runs_dir()/cache``.

    The cache file name holds W and the size and modification time of the cells table, so a rebuilt table is never
    answered from an old cache. A failed cache write is reported on stderr and the table is still returned.
    """
    src = Path(cells_path) if cells_path is not None else tables_dir() / "cells.parquet"
    cpath = None
    if cache:
        st = src.stat()
        cpath = runs_dir() / "cache" / f"history_W{W}_{st.st_size}_{int(st.st_mtime)}.parquet"
        if cpath.exists():
            return pd.read_parquet(cpath)
    cells = pd.read_parquet(src, columns=["track_id", "frame_index", *BASE_FEATURES])
    df = build_history_features(cells, W)
    if cpath is not None:
        # written whole, then renamed, so a partial file is never read; one name per writer, as parallel jobs share it
        tmp = cpath.with_name(f"{cpath.name}.{uuid.uuid4().hex}.tmp")
        try:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(tmp, index=False)
            tmp.replace(cpath)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            print(f"history features: cache not written ({e})", file=sys.stderr)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the 112 handcrafted history features of every cell.")
    ap.add_argument("--cells", type=Path, help="cells table (.parquet); default: tables/cells.parquet")
    ap.add_argument("--W", type=int, default=HISTORY_FRAMES_W, help="history window in frames (default 40)")
    ap.add_argument("--out", type=Path, help="also write the table here (.parquet)")
    ap.add_argument("--no-cache", action="store_true", help="recompute even if a cached table exists")
    a = ap.parse_args(argv)
    cells = a.cells or tables_dir() / "cells.parquet"
    if not cells.is_file():
        print(f"cells table not found: {cells} (run data-cells first)", file=sys.stderr)
        return 3
    df = history_features(a.W, cells, cache=not a.no_cache)
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(a.out, index=False)
    print(f"{len(df)} rows x {len(history_columns(df))} history features (W = {a.W} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
