"""Forecast-row labels, weights and the leave-one-group-out folds on hand-made tables."""

import json

import numpy as np
import pandas as pd
import pytest

from deadahead.data.windows import build_all_windows, build_folds, build_windows, check_folds
from deadahead.paths import REPO_ROOT
from deadahead.survival.config import windows_config

CFG = windows_config()


def movie_table(n_frames=(120, 100)):
    rows = [{"movie_id": f"A_bp_{i + 1}", "cell_line": "A", "treatment": "bp", "fov": i + 1, "num_frames": n,
             "group": "A_bp"} for i, n in enumerate(n_frames)]
    return pd.DataFrame(rows)


def event_table():
    # cell 1 dies of apoptosis at frame 60, cell 2 is alive at the end, cell 3 has an onset after the last frame
    return pd.DataFrame({
        "movie_id": ["A_bp_1", "A_bp_1", "A_bp_2"],
        "track_id": ["A_bp_1:1", "A_bp_1:2", "A_bp_2:3"],
        "event_type": ["APOPTOSIS", "CENSORED", "EXCLUDED"],
        "t_event": [60.0, np.nan, np.nan],
        "label_source": ["curated_label", "type3_no_onset", "excluded_out_of_range"],
    })


def test_labels_and_landmarks():
    w = build_windows(event_table(), movie_table(), CFG, "full_length", None)
    assert set(w["track_id"]) == {"A_bp_1:1", "A_bp_1:2"}          # the out-of-range onset gives no rows
    c1 = w[w["track_id"] == "A_bp_1:1"]
    assert sorted(c1["t_obs_frame"].unique()) == [40, 50]           # entry + W, every 10 frames, before the death
    lab = c1.set_index(["t_obs_frame", "horizon_min"])["label"]
    assert lab[(50, 30)] == "APOPTOSIS"                             # death at 60 in (50, 60]
    assert lab[(40, 30)] == "NO_EVENT" and lab[(40, 60)] == "APOPTOSIS"
    c2 = w[w["track_id"] == "A_bp_1:2"].set_index(["t_obs_frame", "horizon_min"])["label"]
    assert c2[(110, 30)] == "CENSORED_IN_HORIZON"                   # follow-up ends at 119, never a negative
    assert c2[(80, 120)] == "CENSORED_IN_HORIZON" and c2[(70, 120)] == "NO_EVENT"
    assert (w["history_start_frame"] == w["t_obs_frame"] - 40).all()


def test_cut_censors_late_events():
    ev = event_table()
    ev.loc[0, "t_event"] = 110.0
    w = build_windows(ev, movie_table(), CFG, "primary_439", 90)
    c1 = w[w["track_id"] == "A_bp_1:1"]
    assert set(c1["track_terminal_event"]) == {"CENSORED"}
    assert c1["t_obs_frame"].max() == 80                            # last landmark before the cut at 90
    assert np.allclose(c1.groupby("t_obs_frame")["time_to_event_min"].first().loc[80], 30.0)


def test_weights_sum_to_one_per_movie():
    w = build_all_windows(event_table(), movie_table(), CFG)
    tot = w.groupby(["panel", "horizon_min", "movie_id"])["window_weight"].sum()
    assert np.allclose(tot, 1.0)
    per_track = w.groupby(["panel", "horizon_min", "track_id"])["window_weight"].sum()
    assert per_track.nunique() == 1                                 # the two tracks of movie 1 weigh the same


def full_movie_table():
    rows = [{"movie_id": f"{line}_{t}_{f}", "group": f"{line}_{t}"}
            for line in ("DU145", "LNCaP", "PNT1A") for t in ("bp", "do", "st") for f in range(1, 8)]
    return pd.DataFrame(rows)


def test_folds_match_the_released_splits():
    movies = full_movie_table()
    folds = build_folds(movies)
    check_folds(folds, movies)
    assert folds == json.loads((REPO_ROOT / "splits" / "folds.json").read_text())


def test_fold_check_rejects_a_leak():
    movies = full_movie_table()
    folds = build_folds(movies)
    folds["folds"][1]["train_movies"].append(folds["folds"][1]["test_movies"][0])
    with pytest.raises(ValueError):
        check_folds(folds, movies)
