"""Data preparation end to end on two synthetic movies: movie table, tables, frame measures, forecast rows."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml

from deadahead.data import inventory, tables, windows
from deadahead.features import frame_measures
from deadahead.features.history import BASE_FEATURES
from deadahead.paths import REPO_ROOT

_spec = importlib.util.spec_from_file_location("micro_movie", Path(__file__).parent / "data" / "micro_movie.py")
micro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(micro)

PX_UM = 0.629


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Run the four data commands on the synthetic movies; return the raw folder and the tables folder."""
    root = tmp_path_factory.mktemp("micro")
    raw, out = root / "raw", root / "tables"
    micro.write_micro_movies(raw)
    (raw / "QPI_annotated_timelapse_MICRO.zip").write_bytes(b"")  # a downloaded archive is skipped

    # a short landmark grid, so that 20-frame movies have forecast rows: W = 4, one landmark every 2 frames,
    # horizons of 2 and 4 frames, movies cut at frame 15 (1-based)
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "windows.yaml").read_text())
    cfg["landmark"].update(history_frames_W=4, grid_every_frames=2, min_followup_frames=1)
    cfg["horizons_min"] = [6, 12]
    cfg["truncation"]["primary_truncation_frame"] = 15
    config = root / "windows.yaml"
    config.write_text(yaml.safe_dump(cfg))

    table = out / "movies.csv"
    assert inventory.main(["--raw-dir", str(raw), "--out", str(table), "--any-size"]) == 0
    common = ["--movie-table", str(table), "--raw-dir", str(raw), "--config", str(config), "--out-dir", str(out)]
    assert tables.main(common) == 0
    assert frame_measures.main(common) == 0
    assert windows.main(["--tables-dir", str(out), "--config", str(config)]) == 0
    return raw, out


def test_movie_table_paths_are_relative(built):
    raw, out = built
    inv = pd.read_csv(out / "movies.csv")
    assert list(inv["movie_id"]) == ["MICRO_xx_1", "MICRO_xx_2"]
    for c in ("image_path", "mask_path", "labels_path"):
        assert all(not Path(p).is_absolute() and (raw / p).is_file() for p in inv[c])


def test_full_dataset_check_rejects_a_subset(built):
    raw, out = built
    assert inventory.main(["--raw-dir", str(raw), "--out", str(out / "rejected.csv")]) == 2
    assert not (out / "rejected.csv").exists()


def test_missing_file_is_reported(tmp_path):
    micro.write_micro_movies(tmp_path)
    (tmp_path / "QPI_annotated_timelapse_MICRO" / "mask_MICRO_xx_2.tif").unlink()
    df, problems = inventory.find_movies(tmp_path)
    assert list(df["movie_id"]) == ["MICRO_xx_1"]
    assert problems["incomplete"] == ["MICRO_xx_2: missing mask"]


def test_movies_tracks_events(built):
    _, out = built
    movies = pd.read_parquet(out / "movies.parquet")
    assert movies[["num_frames", "height_px", "width_px"]].drop_duplicates().values.tolist() == [[20, 64, 64]]
    assert set(movies["group"]) == {"MICRO_xx"}
    assert not movies["morphology_excluded"].any()

    ev = pd.read_parquet(out / "events.parquet").set_index("track_id")
    assert len(ev) == 12
    one = ev.loc[[f"MICRO_xx_1:{c}" for c in (1, 2, 3, 4, 6, 7)]]
    assert list(one["event_type"]) == ["CENSORED", "APOPTOSIS", "CENSORED", "LYTIC", "CENSORED", "CENSORED"]
    assert one.loc["MICRO_xx_1:2", "t_event"] == 11.0          # 1-based onset 12 -> frame 11
    assert one.loc["MICRO_xx_1:2", "death_frame_raw"] == 12.0
    assert one.loc["MICRO_xx_1:4", "t_event"] == 17.0
    assert one.loc["MICRO_xx_1:1", "t_censor"] == 19.0         # censored at the last frame
    assert one["t_event"].isna().sum() == 4

    tracks = pd.read_parquet(out / "tracks.parquet")
    assert len(tracks) == 12 and (tracks["last_frame"] == 19).all()


def test_frame_measures(built):
    _, out = built
    cells = pd.read_parquet(out / "cells.parquet")
    assert set(BASE_FEATURES) <= set(cells.columns)
    m1 = cells[cells["movie_id"] == "MICRO_xx_1"]
    frames = m1.groupby("cell_num")["frame_index"].agg(["min", "max", "count"])
    assert sorted(frames.index) == [1, 2, 3, 4, 5, 6]              # cell 7 is in no mask; 5 is a daughter
    assert frames.loc[2].tolist() == [0, 11, 12]
    assert frames.loc[6].tolist() == [0, 9, 10]
    assert frames.loc[5].tolist() == [8, 19, 12]
    assert len(cells) == 2 * frames["count"].sum()
    # phase is 3.0 plus the background inside every cell; dry mass = mean phase x area x pixel area
    bg = 0.05 * ((m1["frame_index"] % 3) + 1)
    assert np.allclose(m1["mean_phase"], 3.0 + bg)
    assert np.allclose(m1["dry_mass_pg"], m1["mean_phase"] * m1["area_px"] * PX_UM ** 2)
    assert np.allclose(m1["area_um2"], m1["area_px"] * PX_UM ** 2)
    assert (m1["track_id"] == "MICRO_xx_1:" + m1["cell_num"].astype(str)).all()


def test_forecast_rows(built):
    _, out = built
    w = pd.read_parquet(out / "windows.parquet")
    assert set(w["panel"]) == {"primary_439", "full_length"}
    assert (w["history_end_frame"] - w["history_start_frame"] == 4).all()

    def label(panel, cell, t_obs, h):
        r = w[(w["panel"] == panel) & (w["track_id"] == f"MICRO_xx_1:{cell}")
              & (w["t_obs_frame"] == t_obs) & (w["horizon_min"] == h)]
        assert len(r) == 1
        return r["label"].iloc[0]

    # apoptosis at frame 11: landmarks 4, 6, 8, 10; the 2-frame horizon from 10 holds it, from 8 it does not
    cell2 = w[(w["panel"] == "full_length") & (w["track_id"] == "MICRO_xx_1:2")]
    assert sorted(cell2["t_obs_frame"].unique()) == [4, 6, 8, 10]
    assert label("full_length", 2, 10, 6) == "APOPTOSIS"
    assert label("full_length", 2, 8, 6) == "NO_EVENT"
    assert label("full_length", 2, 8, 12) == "APOPTOSIS"
    # lytic death at frame 17 lies after the cut (frame 14): censored at the cut in the truncated panel
    assert label("primary_439", 4, 12, 6) == "NO_EVENT"
    assert label("primary_439", 4, 12, 12) == "CENSORED_IN_HORIZON"
    assert label("full_length", 4, 16, 6) == "LYTIC"
    assert set(w.loc[(w["panel"] == "primary_439") & (w["track_id"] == "MICRO_xx_1:4"), "track_terminal_event"]) \
        == {"CENSORED"}
    # weights: equal total per track, then normalised to 1 per movie, panel and horizon
    tot = w.groupby(["panel", "horizon_min", "movie_id"])["window_weight"].sum()
    assert np.allclose(tot, 1.0)

    folds = json.loads((out / "folds.json").read_text())
    assert folds["n_folds"] == 1 and folds["folds"][0]["test_movies"] == ["MICRO_xx_1", "MICRO_xx_2"]


def label_rows(rows, movie_id="A_bp_1", n_frames=500):
    """A label table and the movie tables of one synthetic movie, as ``tables.build`` takes them."""
    line, treatment, fov = movie_id.split("_")
    cells = pd.DataFrame(rows)
    cells.insert(0, "movie_id", movie_id)
    cells.insert(1, "cell_line", line)
    cells.insert(2, "treatment", treatment)
    cells.insert(3, "fov", int(fov))
    cells["death_type"] = pd.to_numeric(cells["death_type"]).astype("Int64")
    cells["death_frame"] = pd.to_numeric(cells["death_frame"], errors="coerce")
    scan = pd.DataFrame([{"movie_id": movie_id, "cell_line": line, "treatment": treatment, "fov": int(fov),
                          "n_frames": n_frames, "height": 64, "width": 64}])
    inv = pd.DataFrame([{"movie_id": movie_id, "image_path": "i", "mask_path": "m", "labels_path": "l"}])
    return cells, scan, inv


WINDOWS_CFG = yaml.safe_load((REPO_ROOT / "configs" / "windows.yaml").read_text())


def test_event_types_from_the_spreadsheet_values():
    rows = [{"cell_num": 1, "death_type": 1, "death_frame": 500},     # the last frame, 1-based: inside the movie
            {"cell_num": 2, "death_type": 2, "death_frame": 501},     # after the last frame: EXCLUDED, route kept
            {"cell_num": 3, "death_type": 3, "death_frame": 100},     # a death without a route
            {"cell_num": 4, "death_type": 3, "death_frame": None}]    # alive at the end
    _, _, ev = tables.build(*label_rows(rows), WINDOWS_CFG)
    assert list(ev["event_type"]) == ["APOPTOSIS", "EXCLUDED", "UNTYPED_DEATH", "CENSORED"]
    assert list(ev["death_type_raw"]) == [1, 2, 3, 3]
    assert ev["t_event"].iloc[0] == 499.0 and ev["t_censor"].iloc[3] == 499.0
    assert np.isnan(ev["t_event"].iloc[1]) and ev["death_frame_raw"].iloc[1] == 501.0
    tables.check_events(ev)


@pytest.mark.parametrize("row", [{"cell_num": 1, "death_type": 4, "death_frame": 10},
                                 {"cell_num": 1, "death_type": None, "death_frame": None},
                                 {"cell_num": 1, "death_type": 1, "death_frame": None},
                                 {"cell_num": 1, "death_type": 1, "death_frame": 0},
                                 {"cell_num": None, "death_type": 3, "death_frame": None}])
def test_label_check_rejects(row):
    cells, _, _ = label_rows([row])
    with pytest.raises(ValueError, match="invalid label row"):
        tables.check_labels(cells)


def test_label_check_accepts_every_legal_row():
    cells, _, _ = label_rows([{"cell_num": 1, "death_type": 1, "death_frame": 1},
                              {"cell_num": 2, "death_type": 2, "death_frame": 900},
                              {"cell_num": 3, "death_type": 3, "death_frame": 40},
                              {"cell_num": 4, "death_type": 3, "death_frame": None}])
    tables.check_labels(cells)


def test_spreadsheets_are_checked_when_read(tmp_path):
    micro.write_micro_movies(tmp_path)
    inv, _ = inventory.find_movies(tmp_path)
    tables.read_all_labels(inv, tmp_path)
    bad = pd.DataFrame([{"cell_num": 1, "death_type": 4, "death_frame": 10}])
    bad.to_excel(tmp_path / inv["labels_path"].iloc[1], index=False)
    with pytest.raises(ValueError, match="MICRO_xx_2"):
        tables.read_all_labels(inv, tmp_path)


@pytest.mark.parametrize("onset", [-1.0, 500.0])
def test_event_check_rejects_an_onset_outside_the_movie(onset):
    ev = pd.DataFrame({"t_onset_lower": [onset], "t_event": [onset], "t_onset_upper": [onset], "num_frames": [500]})
    with pytest.raises(ValueError, match="outside its movie"):
        tables.check_events(ev)


def test_event_check_rejects_a_disordered_interval():
    ev = pd.DataFrame({"t_onset_lower": [12.0], "t_event": [10.0], "t_onset_upper": [14.0], "num_frames": [500]})
    with pytest.raises(ValueError, match="out of order"):
        tables.check_events(ev)


def test_excluded_movie_is_flagged_in_movies_and_cells(tmp_path):
    rows = [{"cell_num": 1, "death_type": 3, "death_frame": None}]
    movies, _, _ = tables.build(*label_rows(rows, movie_id="DU145_st_2"), WINDOWS_CFG)
    assert movies["morphology_excluded"].tolist() == [True]

    micro.write_micro_movies(tmp_path)
    folder = tmp_path / "QPI_annotated_timelapse_MICRO"
    args = (str(folder / "QPI_MICRO_xx_1.tif"), str(folder / "mask_MICRO_xx_1.tif"))
    assert frame_measures.extract_movie(*args, "DU145_st_2", PX_UM, 3.0)["morphology_excluded"].all()
    assert not frame_measures.extract_movie(*args, "DU145_st_1", PX_UM, 3.0)["morphology_excluded"].any()


def test_stack_mismatch_is_refused(tmp_path):
    micro.write_micro_movies(tmp_path)
    inv, _ = inventory.find_movies(tmp_path)
    tifffile.imwrite(tmp_path / inv["mask_path"].iloc[0], micro.stacks()[1][:-1])
    with pytest.raises(ValueError, match="differ"):
        tables.scan_movies(inv, tmp_path)


def test_stray_files_are_skipped_and_misnamed_movie_files_reported(tmp_path):
    micro.write_micro_movies(tmp_path)
    (tmp_path / ".DS_Store").write_bytes(b"")
    (tmp_path / "README.txt").write_text("notes")
    (tmp_path / "QPI_MICRO_xx_3_copy.tif").write_bytes(b"")
    df, problems = inventory.find_movies(tmp_path)
    assert len(df) == 2
    assert problems["unmatched"] == ["QPI_MICRO_xx_3_copy.tif"]


def test_unknown_movie_is_refused(built, capsys):
    raw, out = built
    assert frame_measures.main(["--movie-table", str(out / "movies.csv"), "--raw-dir", str(raw), "--out-dir", str(out),
                                "--movies", "MICRO_xx_1", "MICRO_xx_9"]) == 2
    assert "MICRO_xx_9" in capsys.readouterr().err
