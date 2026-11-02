import json

import numpy as np
import pandas as pd
import pytest

from deadahead.evaluation.aggregate import AHEAD_X, AHEAD_Y, NOT_IMPORTANT, paired
from deadahead.evaluation.rows import GROUPS, SEEDS, inner_groups
from deadahead.reproduce.bundle import load_bundle, sha256_file
from deadahead.reproduce.image_warnings import (IMAGE, fig2_csv, image_ref_scores, index_image_predictions,
                                                part_image_warnings)
from deadahead.reproduce.main import PARTS


def test_outcome_labels_use_the_paper_wording():
    for label in (AHEAD_X, AHEAD_Y, NOT_IMPORTANT):
        assert "practically" not in label
    x = {g: 0.5 for g in GROUPS}
    y = {g: 0.4 for g in GROUPS}
    assert paired(x, y)["outcome"] == AHEAD_X == "x ahead (meets the decision rule)"
    assert paired(x, x)["outcome"] == NOT_IMPORTANT == "below the decision rule"


def test_inner_groups_are_the_other_eight_in_fold_order():
    for g in GROUPS:
        hs = inner_groups(g)
        assert len(hs) == 8 and g not in hs and hs == [h for h in GROUPS if h != g]
    with pytest.raises(ValueError):
        inner_groups("unknown")


def _rows():
    out = []
    for g in GROUPS:
        for t in range(4):
            out.append({"movie_id": f"{g}_m", "track_id": f"{g}_m:1", "t_obs_frame": 10 * t, "group": g})
    return pd.DataFrame(out)


def _inner(rows):
    blocks = []
    for g in GROUPS:
        for s in SEEDS:
            for h in inner_groups(g):
                rh = rows[rows["group"] == h]
                for c in ("APOPTOSIS", "LYTIC"):
                    blocks.append(pd.DataFrame({
                        "outer_group": g, "seed": s, "inner_group": h, "cause": c,
                        "movie_id": rh["movie_id"].to_numpy(), "track_id": rh["track_id"].to_numpy(),
                        "t_obs_frame": rh["t_obs_frame"].to_numpy(),
                        "score": GROUPS.index(h) + rh["t_obs_frame"].to_numpy() / 100.0 + (c == "LYTIC")}))
    return pd.concat(blocks, ignore_index=True)


def test_image_reference_scores_pool_the_training_groups_reference_rows():
    rows = _rows()
    bundle = {"predictions": {IMAGE: pd.DataFrame(columns=["group"])}, "inner_predictions_by_method":
              {IMAGE: _inner(rows)}}
    index_image_predictions(bundle)
    ref = (rows["t_obs_frame"] < 20).to_numpy()
    g, s = GROUPS[2], SEEDS[1]
    got = image_ref_scores(bundle, g, s, rows, ref, "LYTIC")
    want = np.concatenate([[GROUPS.index(h) + 0.0 + 1, GROUPS.index(h) + 0.1 + 1] for h in inner_groups(g)])
    assert np.array_equal(got, want)


def test_bundle_reads_the_inner_predictions_of_other_methods(tmp_path):
    rows = _rows().assign(label="NO_EVENT", horizon_min=30, window_weight=1.0)
    rows.to_parquet(tmp_path / "rows.parquet", index=False)
    pd.DataFrame({"track_id": ["x"], "event_type": ["NO_EVENT"], "t_event": [np.nan]}).to_parquet(
        tmp_path / "events.parquet", index=False)
    (tmp_path / "inner_predictions").mkdir()
    _inner(_rows()).to_parquet(tmp_path / "inner_predictions" / "img.parquet", index=False)
    files = {"rows.parquet": {}, "events.parquet": {},
             "inner_predictions/img.parquet": {"method": IMAGE}}
    for rel, info in files.items():
        info["sha256"] = sha256_file(tmp_path / rel)
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"files": files}))
    b = load_bundle(tmp_path)
    assert set(b["inner_predictions_by_method"]) == {IMAGE}
    assert len(b["inner_predictions_by_method"][IMAGE]) == 9 * 3 * 8 * 2 * 4


def test_part_needs_the_image_inner_predictions():
    with pytest.raises(ValueError):
        part_image_warnings({"predictions": {}})


def test_image_warnings_is_a_part():
    assert "image-warnings" in PARTS


def test_fig2_csv_has_the_columns_of_the_figure_table():
    sus = {X: {"nine_folds": {"macro": 0.5}} for X in ("30", "60", "90", "120", "180", "240")}
    by_rate = {r: {"sustained": sus} for r in ("0.2_per_cell_hour", "0.4_per_cell_hour")}
    models = {c: by_rate for c in ("LYTIC", "APOPTOSIS")}
    warn = {"models": {"deadahead": models, "elapsed_time": models}}
    image = {"models": {IMAGE: models}}
    from pathlib import Path
    ref = (Path(__file__).resolve().parents[1] / "docs" / "figures" / "data" / "fig2_warnings.csv").read_text()
    text = fig2_csv(warn, image).splitlines()
    assert text[0] == ref.splitlines()[0]
    assert text[1] == "30," + ",".join(["50.0"] * 10) and len(text) == 7
    assert "dinov2" not in fig2_csv(warn).splitlines()[0]


def test_supplementary_config_matches_the_code():
    from pathlib import Path

    import yaml

    from deadahead.evaluation import alarms as W
    from deadahead.reproduce import gap as G
    from deadahead.reproduce import image_warnings as I
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "evaluation.yaml").read_text())
    assert cfg["image_warnings"]["method"] == I.IMAGE
    assert cfg["image_warnings"]["inner_heads_per_outer_unit"] == len(inner_groups(GROUPS[0]))
    assert str(cfg["image_warnings"]["last_hour_x_min"]) == I.LAST_HOUR
    assert (cfg["gap"]["early_min"], cfg["gap"]["late_min"]) == (G.H_EARLY, G.H_LATE)
    assert tuple(cfg["gap"]["methods"]) == G.METHOD_ORDER
    assert cfg["sensitivity"]["table1_window_end_frame"] == W.CUT_FRAME
    assert cfg["sensitivity"]["landmark_cut_frame"] == W.LANDMARK_CUT


def test_gap_sets_partition_the_rows():
    from deadahead.reproduce.gap import SET_NAMES, gap_sets
    T = np.array([30.0, 90.0, 90.0, 200.0, 50.0, 100.0])
    delta = np.array([2, 2, 1, 0, 0, 0])
    C = np.array([500.0, 500.0, 500.0, 200.0, 50.0, 100.0])
    s = gap_sets(T, delta, C, "LYTIC")
    k = np.stack([s[x] for x in SET_NAMES]).astype(int).sum(axis=0)
    assert (k == 1).all()
    assert s["E"].tolist() == [True, False, False, False, False, False]
    assert s["P"].tolist() == [False, True, False, False, False, False]
    assert s["Q"].tolist() == [False, False, True, False, False, False]
    assert s["N"].tolist() == [False, False, False, True, False, False]
    assert s["U"].tolist() == [False, False, False, False, True, True]


def test_gap_primary_score_is_the_increment_of_the_cumulative_incidence():
    from deadahead.reproduce.gap import given_survival_score, primary_score
    from deadahead.reproduce.warning_times import TIME
    preds = {60: {"LYTIC": np.array([0.1, 0.2]), "APOPTOSIS": np.array([0.1, 0.8])},
             120: {"LYTIC": np.array([0.3, 0.5]), "APOPTOSIS": np.array([0.2, 0.9])}}
    assert np.allclose(primary_score("deadahead", preds, "LYTIC"), [0.2, 0.3])
    assert np.allclose(primary_score(TIME, preds, "LYTIC"), [0.3, 0.5])
    g = given_survival_score("deadahead", preds, "LYTIC")
    assert np.isclose(g[0], 0.2 / 0.8) and np.isnan(g[1])
    assert given_survival_score(TIME, preds, "LYTIC") is None
