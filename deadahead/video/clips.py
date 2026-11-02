"""Clip index for the video model: 16 distinct frames over the past 120 min (40 frames) per landmark.

A clip is stored as an index, not as pixels: for each landmark, the 16 row numbers into its movie's
shard of the all-frames crop store (``deadahead.video.crops``). Frames sit at
``t - round(40 * i / 15)`` for i = 0..15, i.e. ``t - {40, 37, 35, ..., 3, 0}``, oldest first, all inside
the 40-frame history window the handcrafted features also read.

A training fold's index holds only that fold's eight training groups; the held-out group contributes
no pixels, to the adaptation or to the fine-tuning. The builder never reads an outcome column.

    deadahead video-clips [--folds DU145_bp ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from deadahead import paths

#: The forecast-row panel used for every result (one row per landmark and horizon).
PANEL = "primary_439"

#: History window in frames (3 min per frame, so 120 min). Checked against the data.
HISTORY_FRAMES = 40

#: Frames per clip.
CLIP_FRAMES = 16

#: Columns the builder reads; ``label`` is not among them.
INDEX_COLUMNS = ("movie_id", "track_id", "t_obs_frame", "panel", "group",
                 "history_start_frame", "history_end_frame")

#: A column whose name contains one of these may carry the outcome and is refused.
FORBIDDEN_SUBSTRINGS = ("label", "event", "censor", "time_to", "terminal", "weight")


class ClipBuildError(RuntimeError):
    """Raised when a clip index cannot be built as specified."""


def clip_offsets(n_frames: int = CLIP_FRAMES, span: int = HISTORY_FRAMES) -> list[int]:
    """Frame offsets ``round(span * i / (n - 1))``, sorted; subtracted from t, oldest first."""
    if n_frames < 2:
        raise ClipBuildError(f"a clip needs at least 2 frames, got {n_frames}")
    offs = sorted({int(round(span * i / (n_frames - 1))) for i in range(n_frames)})
    if len(offs) != n_frames:
        raise ClipBuildError(
            f"span {span} cannot give {n_frames} distinct frames: {offs} has {len(offs)}")
    return offs                      # [0, 3, 5, ..., 40]; subtracted from t below


def assert_no_label_columns(columns) -> None:
    """Raise if any column could carry the outcome."""
    bad = [c for c in columns
           if any(s in str(c).lower() for s in FORBIDDEN_SUBSTRINGS)]
    if bad:
        raise ClipBuildError(f"outcome columns {bad} may not be read by the clip builder")


def read_landmarks(windows_parquet: Path | str, *, panel: str = PANEL) -> pd.DataFrame:
    """One row per landmark (movie, track, t) of the panel, sorted. No labels."""
    assert_no_label_columns(INDEX_COLUMNS)
    w = pd.read_parquet(windows_parquet, columns=list(INDEX_COLUMNS))
    w = w[w["panel"] == panel]
    if w.empty:
        raise ClipBuildError(f"no rows with panel == {panel!r} in {windows_parquet}")
    span = (w["history_end_frame"] - w["history_start_frame"]).unique()
    if len(span) != 1 or int(span[0]) != HISTORY_FRAMES:
        raise ClipBuildError(
            f"the history window is not a constant {HISTORY_FRAMES} frames: found spans "
            f"{sorted(int(s) for s in span)}")
    if not (w["t_obs_frame"] == w["history_end_frame"]).all():
        raise ClipBuildError("t_obs_frame != history_end_frame for some rows; the offsets assume the "
                             "landmark is the last frame of the window")
    lm = (w.drop_duplicates(["movie_id", "track_id", "t_obs_frame"])
           [["movie_id", "track_id", "t_obs_frame", "group"]]
           .sort_values(["movie_id", "track_id", "t_obs_frame"])
           .reset_index(drop=True))
    return lm


@dataclass(frozen=True)
class FoldClips:
    """One fold's clip index and what had to be dropped."""

    held_out_group: str
    training_groups: tuple[str, ...]
    clips: pd.DataFrame
    landmarks_considered: int
    dropped_missing_frames: int
    movies_used: int
    movies_of_held_out_group: int

    @property
    def retained_fraction(self) -> float:
        n = self.landmarks_considered
        return 0.0 if not n else round(len(self.clips) / n, 4)

    def summary(self) -> dict:
        return {
            "held_out_group": self.held_out_group,
            "training_groups": list(self.training_groups),
            "clips": int(len(self.clips)),
            "landmarks_considered": self.landmarks_considered,
            "dropped_because_a_frame_is_missing_from_the_store": self.dropped_missing_frames,
            "retained_fraction": self.retained_fraction,
            "movies_used": self.movies_used,
            "movies_of_held_out_group_used": self.movies_of_held_out_group,
        }


def frame_rows(shard_index: pd.DataFrame) -> dict[tuple[str, int], int]:
    """Map ``(track_id, frame_index) -> row`` for one movie's shard of the crop store."""
    cols = {"track_id", "frame_index", "row"}
    missing = cols - set(shard_index.columns)
    if missing:
        raise ClipBuildError(f"shard index is missing columns {sorted(missing)}")
    return {(str(t), int(f)): int(r) for t, f, r
            in zip(shard_index["track_id"], shard_index["frame_index"], shard_index["row"])}


def build_fold_index(landmarks: pd.DataFrame, store_dir: Path | str, held_out_group: str, *,
                     offsets: list[int] | None = None) -> FoldClips:
    """One fold's training clip index: the eight training groups, nothing of the held-out group.

    A landmark with a frame missing from the store is dropped and counted. Raises if a training
    group ends up with no clips, so every fold trains on the same group set.
    """
    store_dir = Path(store_dir)
    offs = offsets if offsets is not None else clip_offsets()
    groups = sorted(landmarks["group"].unique())
    if held_out_group not in groups:
        raise ClipBuildError(f"held-out group {held_out_group!r} is not one of {groups}")
    training = tuple(g for g in groups if g != held_out_group)

    train = landmarks[landmarks["group"] != held_out_group]
    # No row of the held-out group may survive the filter.
    leaked = train[train["group"] == held_out_group]
    if len(leaked):
        raise ClipBuildError(f"{len(leaked)} rows of the held-out group {held_out_group} survived the filter")

    rows, dropped = [], 0
    for movie, part in train.groupby("movie_id", sort=True):
        idx_path = store_dir / f"{movie}.parquet"
        if not idx_path.exists():
            raise ClipBuildError(f"the all-frames store has no shard for movie {movie} ({idx_path})")
        lookup = frame_rows(pd.read_parquet(idx_path, columns=["track_id", "frame_index", "row"]))
        for track, t, grp in zip(part["track_id"], part["t_obs_frame"], part["group"]):
            want = [(str(track), int(t) - o) for o in reversed(offs)]   # oldest first
            got = [lookup.get(k) for k in want]
            if any(g is None for g in got):
                dropped += 1
                continue
            rows.append((movie, str(track), int(t), str(grp), np.array(got, dtype=np.int32)))

    if not rows:
        raise ClipBuildError(f"fold {held_out_group}: no landmark had all {len(offs)} frames in the store")
    clips = pd.DataFrame(rows, columns=["movie_id", "track_id", "t_obs_frame", "group", "frame_rows"])
    present = set(clips["group"].unique())
    if present != set(training):
        raise ClipBuildError(
            f"fold {held_out_group}: clips survive for groups {sorted(present)}, not all training "
            f"groups {list(training)}")
    return FoldClips(
        held_out_group=held_out_group, training_groups=training, clips=clips,
        landmarks_considered=int(len(train)), dropped_missing_frames=dropped,
        movies_used=int(clips["movie_id"].nunique()), movies_of_held_out_group=0,
    )


def build_group_index(landmarks: pd.DataFrame, store_dir: Path | str, group: str, *,
                      offsets: list[int] | None = None) -> FoldClips:
    """Clips for one group, for inference only (the held-out group at prediction time).

    Kept apart from ``build_fold_index`` so a training index can never contain the held-out group.
    """
    store_dir = Path(store_dir)
    offs = offsets if offsets is not None else clip_offsets()
    sub = landmarks[landmarks["group"] == group]
    if sub.empty:
        raise ClipBuildError(f"no landmarks in group {group!r}")
    rows, dropped = [], 0
    for movie, part in sub.groupby("movie_id", sort=True):
        idx_path = store_dir / f"{movie}.parquet"
        if not idx_path.exists():
            raise ClipBuildError(f"the all-frames store has no shard for movie {movie} ({idx_path})")
        lookup = frame_rows(pd.read_parquet(idx_path, columns=["track_id", "frame_index", "row"]))
        for track, t in zip(part["track_id"], part["t_obs_frame"]):
            got = [lookup.get((str(track), int(t) - o)) for o in reversed(offs)]
            if any(g is None for g in got):
                dropped += 1
                continue
            rows.append((movie, str(track), int(t), group, np.array(got, dtype=np.int32)))
    if not rows:
        raise ClipBuildError(f"group {group}: no landmark had all {len(offs)} frames in the store")
    clips = pd.DataFrame(rows, columns=["movie_id", "track_id", "t_obs_frame", "group", "frame_rows"])
    return FoldClips(held_out_group=group, training_groups=(), clips=clips,
                     landmarks_considered=int(len(sub)), dropped_missing_frames=dropped,
                     movies_used=int(clips["movie_id"].nunique()),
                     movies_of_held_out_group=int(clips["movie_id"].nunique()))


def write_fold_index(out_dir: Path | str, fold: FoldClips) -> tuple[Path, Path]:
    """Write ``clips__<group>.parquet`` and its JSON summary. Returns both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq = out_dir / f"clips__{fold.held_out_group}.parquet"
    df = fold.clips.copy()
    df["frame_rows"] = df["frame_rows"].map(lambda a: np.asarray(a, dtype=np.int32).tolist())
    df.to_parquet(pq, index=False)
    js = out_dir / f"clips__{fold.held_out_group}.json"
    js.write_text(json.dumps(fold.summary(), indent=2) + "\n")
    return pq, js


class ClipGatherer:
    """Read clips from the all-frames crop store through memory maps (numpy, no torch).

    Several data-loading processes can share one page cache; the store is read-only.
    """

    def __init__(self, store_dir: Path | str, *, dtype=np.float16):
        self.store_dir = Path(store_dir)
        self.dtype = dtype
        self._maps: dict[str, np.ndarray] = {}

    def _shard(self, movie: str) -> np.ndarray:
        m = self._maps.get(movie)
        if m is None:
            p = self.store_dir / f"{movie}.npy"
            if not p.exists():
                raise ClipBuildError(f"the all-frames store has no pixels for movie {movie} ({p})")
            m = np.load(p, mmap_mode="r")
            self._maps[movie] = m
        return m

    def clip(self, movie: str, frame_rows) -> np.ndarray:
        """One clip as ``(frames, H, W)``, oldest frame first, in the store's dtype."""
        rows = np.asarray(frame_rows, dtype=np.int64)
        shard = self._shard(movie)
        if rows.min() < 0 or rows.max() >= len(shard):
            raise ClipBuildError(
                f"clip rows {rows.min()}..{rows.max()} fall outside shard {movie} "
                f"({len(shard)} rows); the index and the store disagree")
        return np.asarray(shard[rows], dtype=self.dtype)

    def batch(self, movies, frame_rows_list) -> np.ndarray:
        """A batch as ``(n, frames, H, W)``."""
        return np.stack([self.clip(m, r) for m, r in zip(movies, frame_rows_list)])

    def close(self) -> None:
        self._maps.clear()


# ------------------------------------------------------------------ default locations
def default_windows() -> Path:
    return paths.tables_dir() / "windows.parquet"


def default_store() -> Path:
    return paths.crops_dir() / "frames128_all"


def default_clips_dir() -> Path:
    return paths.runs_dir() / "videomae" / "clips"


def unknown_movies(asked, known) -> bool:
    """True, with a message, if an asked movie id is not among the known ones."""
    unknown = sorted(set(asked) - set(known))
    if unknown:
        print(f"unknown movie(s) {unknown}", file=sys.stderr)
    return bool(unknown)


def input_missing(path: Path | str, command: str) -> bool:
    """True if the file is absent, after saying which ``deadahead`` command builds it."""
    if Path(path).is_file():
        return False
    print(f"{path} not found; run deadahead {command} first", file=sys.stderr)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the per-fold clip indexes of the video model.")
    ap.add_argument("--windows", type=Path, default=None, help="forecast-row table (default: tables/windows.parquet)")
    ap.add_argument("--store", type=Path, default=None, help="all-frames crop store (default: crops/frames128_all)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--folds", nargs="*", default=None, help="held-out groups to build (default: all nine)")
    ap.add_argument("--force", action="store_true", help="rebuild folds already on disk")
    a = ap.parse_args(argv)
    windows = a.windows or default_windows()
    store = a.store or default_store()
    out_dir = a.out_dir or default_clips_dir()
    if input_missing(windows, "data-windows"):
        return 3

    lm = read_landmarks(windows, panel=PANEL)
    groups = sorted(lm["group"].unique())
    folds = a.folds or groups
    unknown = [f for f in folds if f not in groups]
    if unknown:
        print(f"unknown fold(s) {unknown}; groups are {groups}", file=sys.stderr)
        return 2
    shards = sorted(p.stem for p in Path(store).glob("*.parquet"))
    if len(shards) < lm["movie_id"].nunique():
        print(f"the store has {len(shards)} shards for {lm['movie_id'].nunique()} movies; build it first "
              f"(deadahead.video.crops)", file=sys.stderr)
        return 3
    print(f"{len(lm)} landmarks across {len(groups)} groups, {lm['movie_id'].nunique()} movies", flush=True)
    for g in folds:
        pq = Path(out_dir) / f"clips__{g}.parquet"
        js = Path(out_dir) / f"clips__{g}.json"
        if pq.exists() and js.exists() and not a.force:
            print(f"  {g}: on disk, skipped", flush=True)
            continue
        fold = build_fold_index(lm, store, g)
        write_fold_index(out_dir, fold)
        print(f"  {g}: {len(fold.clips)} clips from {fold.movies_used} movies, "
              f"{fold.dropped_missing_frames} landmarks dropped for a missing frame", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
