import json

import numpy as np
import pandas as pd
import pytest

from deadahead.evaluation.aggregate import AHEAD_X
from deadahead.evaluation.rows import GROUPS, table1_rows
from deadahead.evaluation.rows import SEEDS as R_SEEDS
from deadahead.reproduce.bundle import load_bundle, sha256_file
from deadahead.reproduce.main import main

SEEDS = R_SEEDS


def synthetic_rows(rng) -> pd.DataFrame:
    """Nine groups, two movies each, six cells, 12 frames; one untyped-death cell in the first group."""
    out = []
    for g in GROUPS:
        for mv in range(2):
            movie = f"{g}_{mv}"
            for tr in range(6):
                fate = ["APOPTOSIS", "LYTIC", "NO_EVENT", "NO_EVENT", "LYTIC", "CENSORED_IN_HORIZON"][tr]
                for t in range(0, 120, 10):
                    lab = fate if t >= 80 else "NO_EVENT"
                    out.append({"movie_id": movie, "track_id": f"{movie}__{tr}", "t_obs_frame": t, "group": g,
                                "label": lab, "horizon_min": 30, "window_weight": 1.0})
    rows = pd.DataFrame(out)
    first = rows["track_id"] == f"{GROUPS[0]}_0__3"
    rows.loc[first & (rows["t_obs_frame"] >= 80), "label"] = "UNTYPED_DEATH"
    return rows.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)


def predictions(rows: pd.DataFrame, rng, perfect: bool) -> pd.DataFrame:
    blocks = []
    for g in GROUPS:
        rg = rows[rows["group"] == g]
        for s in SEEDS:
            for c in ("APOPTOSIS", "LYTIC"):
                score = (rg["label"] == c).to_numpy(float) if perfect else rng.random(len(rg))
                blocks.append(pd.DataFrame({"group": g, "seed": s, "cause": c, "movie_id": rg["movie_id"].to_numpy(),
                                            "track_id": rg["track_id"].to_numpy(),
                                            "t_obs_frame": rg["t_obs_frame"].to_numpy(), "score": score}))
    return pd.concat(blocks, ignore_index=True)


def synthetic_events(rows: pd.DataFrame) -> pd.DataFrame:
    """Typed deaths at frame 115 (rows are 10 frames = 30 min apart, so each death has one row in each 30-min bin)."""
    ev = rows.drop_duplicates("track_id")[["track_id"]].copy()
    fate = ev["track_id"].str.split("__").str[1].astype(int).map(
        {0: "APOPTOSIS", 1: "LYTIC", 2: "NONE", 3: "NONE", 4: "LYTIC", 5: "NONE"})
    ev["event_type"] = fate.to_numpy()
    ev.loc[ev["track_id"] == f"{GROUPS[0]}_0__3", "event_type"] = "UNTYPED_DEATH"
    ev["t_event"] = np.where(ev["event_type"] != "NONE", 115.0, np.nan)
    return ev.reset_index(drop=True)


def inner_predictions(rows: pd.DataFrame, rng) -> pd.DataFrame:
    blocks = []
    for g in GROUPS:
        for s in SEEDS:
            for h in [x for x in GROUPS if x != g]:
                rh = rows[rows["group"] == h]
                for c in ("APOPTOSIS", "LYTIC"):
                    blocks.append(pd.DataFrame({"outer_group": g, "seed": s, "inner_group": h, "cause": c,
                                                "movie_id": rh["movie_id"].to_numpy(),
                                                "track_id": rh["track_id"].to_numpy(),
                                                "t_obs_frame": rh["t_obs_frame"].to_numpy(),
                                                "score": rng.random(len(rh)) * 0.5}))
    return pd.concat(blocks, ignore_index=True)


def write_bundle(path, rows, preds: dict, inner=None):
    files = {}
    rows.to_parquet(path / "rows.parquet", index=False)
    ev = synthetic_events(rows)
    ev.to_parquet(path / "events.parquet", index=False)
    files["rows.parquet"] = {"rows": len(rows)}
    files["events.parquet"] = {"rows": len(ev)}
    if inner is not None:
        mv = pd.DataFrame({"movie_id": sorted(rows["movie_id"].unique()), "num_frames": 120})
        mv.to_parquet(path / "movies.parquet", index=False)
        inner.to_parquet(path / "inner_predictions.parquet", index=False)
        files["movies.parquet"] = {"rows": len(mv)}
        files["inner_predictions.parquet"] = {"rows": len(inner)}
    (path / "predictions").mkdir()
    for m, df in preds.items():
        rel = "predictions/" + m.replace("|", "__") + ".parquet"
        df.to_parquet(path / rel, index=False)
        files[rel] = {"method": m, "rows": len(df)}
    for rel in files:
        files[rel]["sha256"] = sha256_file(path / rel)
    (path / "MANIFEST.json").write_text(json.dumps({"files": files}))


@pytest.fixture()
def bundle(tmp_path):
    rng = np.random.default_rng(7)
    rows = synthetic_rows(rng)
    write_bundle(tmp_path, rows, {"deadahead": predictions(rows, rng, True),
                                  "videomae_finetuned": predictions(rows, rng, False),
                                  "handcrafted|refit": predictions(rows, rng, False)},
                 inner=inner_predictions(rows, rng))
    return tmp_path


def test_cohort_removes_the_untyped_cell_and_counts_u():
    rows = synthetic_rows(np.random.default_rng(0))
    tr = table1_rows(rows, {f"{GROUPS[0]}_0__3"})
    assert tr["counts"]["cohort_removed"]["tracks"] == 1
    assert tr["counts"]["cohort_removed"]["landmarks"] == 12
    assert tr["counts"]["original_untyped_windows_scored_negative"] == 4
    assert tr["counts"]["U_excluded"] == 9 * 2 * 4
    assert not (tr["original"]["label"] == "UNTYPED_DEATH").any()


def test_reproduce_runs_on_a_synthetic_bundle(bundle, capsys):
    out = bundle / "t1.json"
    assert main(["--bundle", str(bundle), "--part", "table1", "--out", str(out)]) == 0
    res = json.loads(out.read_text())["table1"]
    coh = res["cohort"]
    assert set(coh["methods"]) == {"deadahead", "videomae_finetuned", "handcrafted|refit", "elapsed_time"}
    for c in ("APOPTOSIS", "LYTIC"):
        assert coh["methods"]["deadahead"][c]["nine_folds"]["macro"] == 1.0
        assert coh["methods"]["deadahead"][c]["nine_folds"]["n_groups"] == 9
        k = f"deadahead - handcrafted|refit|{c}"
        assert coh["contrasts"][k]["nine_folds"]["x_ahead"] == 9
        assert coh["contrasts"][k]["nine_folds"]["outcome"] == AHEAD_X
    assert res["counts"]["cohort_removed"]["tracks"] == 1
    assert "Table 1" in capsys.readouterr().out


def test_bundle_manifest_mismatch_is_refused(bundle):
    (bundle / "events.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError):
        load_bundle(bundle)


def test_bundle_size_mismatch_is_reported(bundle):
    from deadahead.reproduce.bundle import check_files, read_manifest
    m = read_manifest(bundle)
    assert check_files(bundle, m) == []
    m["files"]["rows.parquet"]["bytes"] = (bundle / "rows.parquet").stat().st_size + 1
    assert check_files(bundle, m) == ["rows.parquet: size differs from the manifest"]


def test_reproduce_warnings_on_a_synthetic_bundle(bundle, capsys):
    """DeadAhead scores 1 on the rows within 105 min of a typed death (label set from frame 80) and 0 elsewhere;
    its inner scores are below 0.5, so exactly the bins up to 120 min before death are alarmed. The bundle holds
    no image-baseline inner predictions and no survival landmarks, so --part all skips those two parts."""
    out = bundle / "all.json"
    assert main(["--bundle", str(bundle), "--part", "all", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    for p in ("image-warnings", "gap"):
        assert f"skipped {p}: its inputs are not in the bundle" in printed
    full = json.loads(out.read_text())
    assert set(full) == {"data", "table1", "warnings", "sensitivity"}
    fig2 = pd.read_csv(bundle / "fig2_warnings.csv")
    assert fig2["X_min"].tolist() == [30, 60, 90, 120, 180, 240]
    for c in ("apoptosis", "lytic"):
        for r in ("0.2", "0.4"):
            assert fig2[f"deadahead_{c}_rate{r}"].tolist() == [100.0, 100.0, 100.0, 100.0, 0.0, 0.0]
            assert f"elapsed_time_{c}_rate{r}" in fig2
    assert not any(k.startswith("dinov2") for k in fig2)
    res = full["warnings"]
    for c in ("APOPTOSIS", "LYTIC"):
        for rate in ("0.2_per_cell_hour", "0.4_per_cell_hour"):
            sus = res["models"]["deadahead"][c][rate]["sustained"]
            assert [sus[x]["nine_folds"]["macro"] for x in ("30", "60", "90", "120", "180", "240")] == [
                1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
            ach = res["models"]["deadahead"][c][rate]["achieved_rate_equal_group_macro"]["nine_folds"]["macro"]
            assert ach == 0.0
            t = res["models"]["elapsed_time"][c][rate]
            assert t["achieved_rate_exposure_weighted"]["nine_folds"]["rate_per_cell_hour"] >= 0.0
    # reference rows: frames 0..30 of every cell (no death in the next 240 min, followed through it)
    assert res["rows"]["reference_rows"] == 9 * 2 * 6 * 4 - 4
    assert res["rows"]["cohort_removed"]["tracks"] == 1
