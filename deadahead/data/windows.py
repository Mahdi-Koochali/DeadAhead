"""Forecast rows (``windows.parquet``) and the nine leave-one-group-out folds (``folds.json``).

    deadahead data-windows [--tables-dir tables/] [--config configs/windows.yaml] [--folds-out FILE]

A forecast row is one (track, landmark, horizon). Landmarks are every 10 frames along a track, starting once 40
frames of history exist and ending one frame before the track's terminal event or the end of its follow-up. Two
panels are written:
    primary_439   every movie cut at frame 439 (1-based), the length of the shortest group; events after the cut
                  are censored at the cut.
    full_length   the movies at their full length.

Label at (landmark t, horizon h):
    APOPTOSIS / LYTIC / UNTYPED_DEATH   the terminal event falls in (t, t + h]
    NO_EVENT                            the cell is followed event-free through the whole horizon
    CENSORED_IN_HORIZON                 follow-up ends inside the horizon without an event; never a negative
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from deadahead import paths
from deadahead.survival.config import windows_config

TERMINAL = {"APOPTOSIS", "LYTIC", "UNTYPED_DEATH"}

FOLD_UNIT = "cell line x treatment group"
FOLD_SEED = 2601562  # recorded with the folds; the leave-one-group-out folds themselves draw no random numbers


def build_folds(movies: pd.DataFrame) -> dict:
    """Leave-one-group-out folds over the line x treatment groups, in sorted group order."""
    groups = sorted(movies["group"].unique())

    folds = []
    for i, g in enumerate(groups):
        test = sorted(movies.loc[movies["group"] == g, "movie_id"])
        train = sorted(movies.loc[movies["group"] != g, "movie_id"])
        folds.append(
            {
                "fold": i,
                "test_group": g,
                "test_movies": test,
                "train_groups": [x for x in groups if x != g],
                "train_movies": train,
                "n_test_movies": len(test),
                "n_train_movies": len(train),
            }
        )
    return {
        "design": "leave_one_group_out",
        "unit": FOLD_UNIT,
        "n_folds": len(folds),
        "seed": FOLD_SEED,
        "folds": folds,
    }


def check_folds(folds: dict, movies: pd.DataFrame) -> None:
    """Raise unless every movie is held out exactly once and no fold trains on a held-out movie."""
    seen: dict[str, int] = {}
    for f in folds["folds"]:
        for m in f["test_movies"]:
            if m in seen:
                raise ValueError(f"{m} is held out in two folds")
            seen[m] = f["fold"]
        overlap = set(f["test_movies"]) & set(f["train_movies"])
        if overlap:
            raise ValueError(f"fold {f['fold']} trains on held-out movies: {sorted(overlap)}")
    if len(seen) != len(movies):
        raise ValueError(f"{len(seen)} movies held out, {len(movies)} in the movie table")


def build_windows(
    events: pd.DataFrame, movies: pd.DataFrame, cfg: dict, panel: str, cut: int | None
) -> pd.DataFrame:
    """The forecast rows of one panel: one row per (track, landmark, horizon).

    ``cut`` is the last frame (0-based) kept in every movie, or None for the full movies. ``window_weight`` gives
    each track the same total weight and then normalises per movie, per panel and horizon.
    """
    lm = cfg["landmark"]
    W = int(lm["history_frames_W"])
    step = int(lm["grid_every_frames"])
    min_follow = int(lm["min_followup_frames"])
    horizons_frames = [int(h / cfg["frame_interval_min"]) for h in cfg["horizons_min"]]
    n_frames = dict(zip(movies["movie_id"], movies["num_frames"]))

    rows = []
    for ev in events.itertuples(index=False):
        # onsets after the last frame of their movie are left out of the forecast rows
        if ev.label_source == "excluded_out_of_range":
            continue

        last_obs = n_frames[ev.movie_id] - 1
        if cut is not None:
            last_obs = min(last_obs, cut)

        if ev.event_type in TERMINAL and pd.notna(ev.t_event):
            t_e = float(ev.t_event)
            if t_e > last_obs:
                # the event lies after the cut: the cell is censored at the cut in this panel
                t_end, ends_with_event, etype = float(last_obs), False, "CENSORED"
            else:
                t_end, ends_with_event, etype = t_e, True, ev.event_type
        else:
            t_end, ends_with_event, etype = float(last_obs), False, "CENSORED"

        # landmark grid: the first landmark is track entry + W, then one every `step` frames
        first_obs = 0 + W
        last_landmark = t_end - min_follow
        if last_landmark < first_obs:
            continue
        for t_obs in range(first_obs, int(np.floor(last_landmark)) + 1, step):
            for h_min, h_fr in zip(cfg["horizons_min"], horizons_frames):
                h_end = t_obs + h_fr
                if ends_with_event and t_obs < t_end <= h_end:
                    label, within, censored = etype, True, False
                elif t_end >= h_end:
                    # followed event-free through the whole horizon: a negative, even if censored later
                    label, within, censored = "NO_EVENT", False, False
                else:
                    # follow-up ends inside the horizon without an event: never a negative
                    label, within, censored = "CENSORED_IN_HORIZON", False, True
                rows.append(
                    (
                        ev.movie_id,
                        ev.track_id,
                        panel,
                        t_obs,
                        t_obs - W,
                        t_obs,
                        h_min,
                        label,
                        within,
                        censored,
                        (t_end - t_obs) * cfg["frame_interval_min"],
                        etype,
                    )
                )

    w = pd.DataFrame(
        rows,
        columns=[
            "movie_id",
            "track_id",
            "panel",
            "t_obs_frame",
            "history_start_frame",
            "history_end_frame",
            "horizon_min",
            "label",
            "event_within_horizon",
            "censored_in_horizon",
            "time_to_event_min",
            "track_terminal_event",
        ],
    )
    if w.empty:
        return w

    w["observation_time_min"] = w["t_obs_frame"] * cfg["frame_interval_min"]
    w["example_id"] = w["track_id"] + "@" + w["t_obs_frame"].astype(str) + "h" + w["horizon_min"].astype(str)

    # each row weighs 1 / (rows of its track), then the weights are normalised per movie
    per_track = w.groupby(["panel", "horizon_min", "track_id"])["example_id"].transform("count")
    w["window_weight"] = 1.0 / per_track
    movie_tot = w.groupby(["panel", "horizon_min", "movie_id"])["window_weight"].transform("sum")
    w["window_weight"] = w["window_weight"] / movie_tot

    w = w.merge(movies[["movie_id", "group", "cell_line", "treatment", "fov"]], on="movie_id")
    return w


def build_all_windows(events: pd.DataFrame, movies: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Both panels, the truncated one first. Raises ValueError if a history window leaves [0, landmark]."""
    cut = int(cfg["truncation"]["primary_truncation_frame"]) - 1  # 1-based setting -> 0-based frame
    prim = build_windows(events, movies, cfg, "primary_439", cut)
    full = build_windows(events, movies, cfg, "full_length", None)
    w = pd.concat([prim, full], ignore_index=True)
    if not bool((w["history_end_frame"] <= w["t_obs_frame"]).all()):
        raise ValueError("a history window extends past its landmark")
    if not bool((w["history_start_frame"] >= 0).all()):
        raise ValueError("a history window starts before frame 0")
    return w


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write the forecast rows and the leave-one-group-out folds.")
    ap.add_argument("--tables-dir", type=Path, default=None, help="folder with movies and events .parquet "
                    "(default: tables/); windows.parquet is written there")
    ap.add_argument("--config", type=Path, default=None, help="windows settings (default: configs/windows.yaml)")
    ap.add_argument("--folds-out", type=Path, default=None, help="default: <tables-dir>/folds.json")
    a = ap.parse_args(argv)
    d = a.tables_dir or paths.tables_dir()
    for p in (d / "events.parquet", d / "movies.parquet"):
        if not p.is_file():
            print(f"missing {p} (run data-tables first)", file=sys.stderr)
            return 3

    cfg = windows_config(a.config)
    events = pd.read_parquet(d / "events.parquet")
    movies = pd.read_parquet(d / "movies.parquet")

    folds = build_folds(movies)
    check_folds(folds, movies)
    folds_out = a.folds_out or d / "folds.json"
    folds_out.parent.mkdir(parents=True, exist_ok=True)
    folds_out.write_text(json.dumps(folds, indent=1) + "\n")

    w = build_all_windows(events, movies, cfg)
    w.to_parquet(d / "windows.parquet", index=False)
    for panel, sub in w.groupby("panel", sort=False):
        n_lm = sub[["track_id", "t_obs_frame"]].drop_duplicates().shape[0]
        print(f"{panel:<12} {len(sub):>8} rows  {n_lm:>7} landmarks  {sub['track_id'].nunique():>5} tracks")
    print(f"{folds['n_folds']} folds -> {folds_out}")
    print(f"forecast rows -> {d / 'windows.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
