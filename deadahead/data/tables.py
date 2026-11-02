"""Movie, track and event tables of CellDeathDetect, from the movie table, the label spreadsheets and the stack
headers.

    deadahead data-tables [--movie-table tables/movies.csv] [--raw-dir DIR] [--out-dir tables/]

Writes ``movies.parquet``, ``tracks.parquet`` and ``events.parquet``. Frames are counted from 0 in every table; the
spreadsheets count from 1, and their value is kept in ``events.death_frame_raw``. The spreadsheet's death type is kept
in ``events.death_type_raw``, so an EXCLUDED cell keeps its route.

Event types (``death_type`` in the spreadsheets):
    1 -> APOPTOSIS, 2 -> LYTIC (the two competing risks);
    3 with an onset frame -> UNTYPED_DEATH (a death of unknown route);
    3 without an onset frame -> CENSORED at the last frame (alive at the end of the movie, never a negative);
    an onset after the last frame of its movie -> EXCLUDED (kept in the table, left out of the forecast rows).
Any other spreadsheet value stops the build (``check_labels``). Division is not an event. The curated tracks cover the
whole movie, so every censoring is at the end of the movie.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import tifffile

from deadahead import paths
from deadahead.survival.config import windows_config

DATASET_ID = "celldeathdetect"
MODALITY = "QPI"
MORPHOLOGY_EXCLUDED = {"DU145_st_2"}  # its curated outlines failed a visual check


def read_labels(path: Path, movie_id: str, cell_line: str, treatment: str, fov: int) -> pd.DataFrame:
    """One label spreadsheet as a table: movie_id, cell_line, treatment, fov, cell_num, death_type, death_frame."""
    df = pd.read_excel(path, engine="openpyxl")
    norm = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=norm)
    required = {"cell_num", "death_type", "death_frame"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {sorted(missing)}; saw {list(df.columns)}")
    df = df[["cell_num", "death_type", "death_frame"]].copy()
    df.insert(0, "movie_id", movie_id)
    df.insert(1, "cell_line", cell_line)
    df.insert(2, "treatment", treatment)
    df.insert(3, "fov", fov)
    return df


def read_all_labels(inv: pd.DataFrame, raw: Path) -> pd.DataFrame:
    """The label spreadsheets of every movie in ``inv``, one row per annotated cell, sorted by movie and cell.

    ``death_frame`` is the 1-based onset frame (missing for cells without an onset). ``check_labels`` refuses a row
    outside the cases of the module docstring.
    """
    frames = []
    for r in inv.itertuples():
        frames.append(read_labels(raw / r.labels_path, r.movie_id, r.cell_line, r.treatment, int(r.fov)))
    cells = pd.concat(frames, ignore_index=True)
    cells["death_type"] = pd.to_numeric(cells["death_type"], errors="coerce").astype("Int64")
    cells["death_frame"] = pd.to_numeric(cells["death_frame"], errors="coerce")
    check_labels(cells)
    return cells.sort_values(["cell_line", "treatment", "fov", "cell_num"]).reset_index(drop=True)


def check_labels(cells: pd.DataFrame) -> None:
    """Raise unless every row has a cell number, a death type of 1, 2 or 3 and, for types 1 and 2, an onset >= 1."""
    typed = cells["death_type"].isin([1, 2])
    bad = (cells["cell_num"].isna() | ~cells["death_type"].isin([1, 2, 3]) | (cells["death_frame"] < 1)
           | (typed & cells["death_frame"].isna()))
    if bad.any():
        first = cells[bad].iloc[0]
        raise ValueError(f"{int(bad.sum())} invalid label row(s), the first in {first['movie_id']}: "
                         f"cell {first['cell_num']}, death type {first['death_type']}, onset {first['death_frame']}")


def stack_shape(path: Path) -> tuple[int, int, int]:
    """(frames, height, width) of a multi-page TIFF, read from its page headers only."""
    with tifffile.TiffFile(path) as tf:
        p0 = tf.pages[0]
        return len(tf.pages), int(p0.imagelength), int(p0.imagewidth)


def scan_movies(inv: pd.DataFrame, raw: Path) -> pd.DataFrame:
    """movie_id, cell_line, treatment, fov, n_frames, height and width of every movie in ``inv``.

    Raises ValueError when a mask stack and its phase stack differ in frame count or frame size.
    """
    rows = []
    for r in inv.itertuples():
        img = stack_shape(raw / r.image_path)
        msk = stack_shape(raw / r.mask_path)
        if img != msk:
            raise ValueError(f"{r.movie_id}: phase stack {img} and mask stack {msk} differ (frames, height, width)")
        rows.append({"movie_id": r.movie_id, "cell_line": r.cell_line, "treatment": r.treatment, "fov": r.fov,
                     "n_frames": img[0], "height": img[1], "width": img[2]})
    return pd.DataFrame(rows)


def build(labels: pd.DataFrame, scan: pd.DataFrame, inv: pd.DataFrame,
          cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The movies, tracks and events tables.

    ``labels`` comes from ``read_all_labels``, ``scan`` from ``scan_movies``, ``inv`` is the movie table and ``cfg``
    the windows settings (frame interval and pixel size). Tracks and events have one row per annotated cell.
    """
    interval = float(cfg["frame_interval_min"])
    px = float(cfg["pixel_size_um_per_px"])

    # ---------------------------------------------------------------- movies
    movies = (
        scan[["movie_id", "cell_line", "treatment", "fov", "n_frames", "height", "width"]]
        .copy()
        .rename(columns={"n_frames": "num_frames", "height": "height_px", "width": "width_px"})
    )
    movies["dataset_id"] = DATASET_ID
    movies["modality"] = MODALITY
    movies["pixel_size_um"] = px
    movies["frame_interval_min"] = interval
    movies["pixel_size_source"] = "documentation"  # the files carry no acquisition metadata
    movies["frame_interval_source"] = "documentation"
    movies["group"] = movies["cell_line"] + "_" + movies["treatment"]
    movies = movies.merge(
        inv[["movie_id", "image_path", "mask_path", "labels_path"]], on="movie_id", how="left"
    ).rename(columns={"image_path": "source_path"})
    # the movie whose curated outlines failed a visual check; it is left out before any model is scored
    movies["morphology_excluded"] = movies["movie_id"].isin(MORPHOLOGY_EXCLUDED)
    # leave-one-group-out: each movie is held out with its own group
    movies["split"] = "logo:" + movies["group"]

    # ---------------------------------------------------------------- events / tracks
    n_frames = dict(zip(scan["movie_id"], scan["n_frames"]))
    labels = labels.copy()
    labels["n_frames"] = labels["movie_id"].map(n_frames)
    labels["track_id"] = labels["movie_id"] + ":" + labels["cell_num"].astype(str)
    labels["has_onset"] = labels["death_frame"].notna()
    labels["out_of_range"] = labels["has_onset"] & (labels["death_frame"] > labels["n_frames"])

    def classify(r) -> tuple[str, str, str]:
        """(event_type, label_source, end_reason) of one annotated cell."""
        if r["out_of_range"]:
            # an onset after the last frame: kept in the table, left out of the forecast rows
            return ("EXCLUDED", "excluded_out_of_range", "excluded_out_of_range")
        if r["death_type"] == 1:
            return ("APOPTOSIS", "curated_label", "apoptosis")
        if r["death_type"] == 2:
            return ("LYTIC", "curated_label", "lytic")
        if r["death_type"] == 3 and r["has_onset"]:
            # a death was observed but its route is unknown; this is not censoring
            return ("UNTYPED_DEATH", "type3_with_onset", "untyped_death")
        # alive at the last frame: censored, never a negative
        return ("CENSORED", "type3_no_onset", "movie_end")

    cls = labels.apply(classify, axis=1, result_type="expand")
    labels[["event_type", "label_source", "end_reason"]] = cls

    # 0-based frame index; the 1-based spreadsheet value is kept in death_frame_raw
    labels["t_event"] = labels["death_frame"] - 1.0
    labels.loc[~labels["has_onset"] | labels["out_of_range"], "t_event"] = pd.NA

    events = pd.DataFrame(
        {
            "event_id": labels["track_id"],
            "dataset_id": DATASET_ID,
            "movie_id": labels["movie_id"],
            "track_id": labels["track_id"],
            "cell_num": labels["cell_num"],
            "event_type": labels["event_type"],
            "t_last_normal": pd.NA,  # not annotated in the dataset
            "t_onset_lower": labels["t_event"],
            "t_event": labels["t_event"],
            "t_onset_upper": labels["t_event"],
            "t_confirmation": pd.NA,  # not annotated in the dataset
            "onset_source": labels["has_onset"].map({True: "curated", False: "none"}),
            "label_source": labels["label_source"],
            "death_frame_raw": labels["death_frame"],
            "death_type_raw": labels["death_type"],
            "num_frames": labels["n_frames"],
            "confidence": 1.0,
            "notes": labels["out_of_range"].map(
                {True: "death_frame beyond last frame of movie", False: ""}
            ),
        }
    )
    # censoring time of the cells without an event: the last frame of the movie (0-based)
    events["t_censor"] = pd.NA
    is_cens = events["event_type"] == "CENSORED"
    events.loc[is_cens, "t_censor"] = (events.loc[is_cens, "num_frames"] - 1).astype(float)

    # the label set of the curated masks is the same in every frame, so each track spans the whole movie
    tracks = pd.DataFrame(
        {
            "track_id": labels["track_id"],
            "movie_id": labels["movie_id"],
            "cell_num": labels["cell_num"],
            "first_frame": 0,
            "last_frame": labels["n_frames"] - 1,
            "num_observations": labels["n_frames"],
            "fraction_missing": 0.0,
            "end_reason": labels["end_reason"],
            "tracking_confidence": 1.0,
        }
    )
    return movies, tracks, events


def check_events(events: pd.DataFrame) -> None:
    """Raise unless every onset satisfies lower <= event <= upper and lies inside its movie."""
    ordered = events.dropna(subset=["t_event"])
    if not ((ordered["t_onset_lower"] <= ordered["t_event"]).all()
            and (ordered["t_event"] <= ordered["t_onset_upper"]).all()):
        raise ValueError("an onset interval is out of order")
    if not ((ordered["t_event"] >= 0).all() and (ordered["t_event"] <= ordered["num_frames"] - 1).all()):
        raise ValueError("an event time lies outside its movie")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write the movies, tracks and events tables of CellDeathDetect.")
    ap.add_argument("--movie-table", type=Path, default=None, help="movie table (.csv); default: tables/movies.csv")
    ap.add_argument("--raw-dir", type=Path, default=None, help="extracted archives (default: raw/celldeathdetect)")
    ap.add_argument("--config", type=Path, default=None, help="windows settings (default: configs/windows.yaml)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: tables/")
    a = ap.parse_args(argv)
    table = a.movie_table or paths.tables_dir() / "movies.csv"
    raw = a.raw_dir or paths.raw_dir()
    out = a.out_dir or paths.tables_dir()
    if not table.is_file():
        print(f"movie table not found: {table} (run data-inventory first)", file=sys.stderr)
        return 3

    inv = pd.read_csv(table)
    cfg = windows_config(a.config)
    labels = read_all_labels(inv, raw)
    scan = scan_movies(inv, raw)
    movies, tracks, events = build(labels, scan, inv, cfg)
    check_events(events)

    out.mkdir(parents=True, exist_ok=True)
    movies.to_parquet(out / "movies.parquet", index=False)
    tracks.to_parquet(out / "tracks.parquet", index=False)
    events.to_parquet(out / "events.parquet", index=False)
    for k, v in events["event_type"].value_counts().items():
        print(f"  {k:<14} {v:>5}")
    print(f"{len(movies)} movies, {len(tracks)} tracks, {len(events)} events -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
