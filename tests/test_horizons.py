import json

import numpy as np
import pandas as pd
import pytest

from deadahead.evaluation.alarms import NOMINAL, PRIMARY_RATE
from deadahead.evaluation.horizons import fixed_horizon_summary, fixed_horizon_unit, outcome_sets
from deadahead.evaluation.rows import CAUSES, GROUPS, SEEDS, inner_groups
from deadahead.reproduce.bundle import load_bundle, sha256_file
from deadahead.reproduce.gap import DEADAHEAD, DEEPHIT, HANDCRAFTED_SURVIVAL, IMAGE_SURVIVAL
from deadahead.reproduce.horizons import (CIF_MODELS, HORIZONS, SURV_KEYS, cif_preds, horizons_block,
                                          index_horizon_predictions, inner_ref_scores)
from deadahead.reproduce.main import PARTS, available_parts
from deadahead.reproduce.warning_times import TIME, index_predictions

METHODS = (DEADAHEAD, IMAGE_SURVIVAL, HANDCRAFTED_SURVIVAL, DEEPHIT)


def _population(seed=0):
    """Each group: one movie of 160 frames, 8 cells, a landmark every 10 frames until death; deaths alternate
    between the causes, two cells survive the movie."""
    rng = np.random.default_rng(seed)
    out = []
    for g in GROUPS:
        for k in range(8):
            death = None if k >= 6 else int(rng.integers(20, 150))
            cause = 1 + (k % 2)
            end = 160 if k % 3 else 120
            for t in range(0, 110, 10):
                if death is not None and t >= death:
                    break
                C = (end - t) * 3.0
                if death is not None and death <= end:
                    T, d = (death - t) * 3.0, cause
                else:
                    T, d = C, 0
                out.append({"movie_id": f"{g}_m", "track_id": f"{g}_m:{k}", "t_obs_frame": t, "group": g,
                            "T_min": T, "delta": d, "C_min": C})
    return pd.DataFrame(out).sort_values(["movie_id", "track_id", "t_obs_frame"]).reset_index(drop=True)


def _score(pop, h, c, offset=0.0):
    """A cumulative incidence that ranks the deaths of the cause within h above the rest, with noise."""
    rng = np.random.default_rng(int(h) + 7 * (c == "LYTIC"))
    y = ((pop["delta"] == (2 if c == "LYTIC" else 1)) & (pop["T_min"] <= h)).to_numpy()
    return np.clip(0.05 + 0.5 * y + 0.2 * rng.uniform(size=len(pop)) + offset, 0, 1)


def _bundle(pop):
    ci, ci2, inner30, inner_h = [], [], [], []
    for m in METHODS:
        for g in GROUPS:
            pg = pop[pop["group"] == g].reset_index(drop=True)
            for s in SEEDS:
                key = {"method": m, "group": g, "seed": s, "movie_id": pg["movie_id"], "track_id": pg["track_id"],
                       "t_obs_frame": pg["t_obs_frame"]}
                off = 0.01 * METHODS.index(m)
                ci.append(pd.DataFrame({**key, **{f"F_{c}_{h}": _score(pg, h, c, off) for h in (60, 120)
                                                  for c in CAUSES}}))
                ci2.append(pd.DataFrame({**key, **{f"F_{c}_{h}": _score(pg, h, c, off) for h in (30, 240)
                                                   for c in CAUSES}}))
    for g in GROUPS:
        for s in SEEDS:
            for hh in inner_groups(g):
                ph = pop[pop["group"] == hh].reset_index(drop=True)
                key = {"outer_group": g, "seed": s, "inner_group": hh, "movie_id": ph["movie_id"],
                       "track_id": ph["track_id"], "t_obs_frame": ph["t_obs_frame"]}
                for c in CAUSES:
                    inner30.append(pd.DataFrame({**key, "cause": c, "score": _score(ph, 30, c)}))
                inner_h.append(pd.DataFrame({**key, **{f"F_{c}_{h}": _score(ph, h, c) for h in (60, 120, 240)
                                                       for c in CAUSES}}))
    b = {"cumulative_incidence": pd.concat(ci, ignore_index=True),
         "cumulative_incidence_30_240": pd.concat(ci2, ignore_index=True),
         "inner_predictions": pd.concat(inner30, ignore_index=True),
         "inner_cumulative_incidence": pd.concat(inner_h, ignore_index=True),
         "predictions": {DEADAHEAD: pd.DataFrame(columns=["group", "seed", "cause"])}}
    index_predictions(b)
    index_horizon_predictions(b)
    return b


@pytest.fixture(scope="module")
def world():
    pop = _population()
    return pop, _bundle(pop)


def test_fixed_horizon_unit_counts_and_summary():
    T = np.array([10.0, 20.0, 100.0, 100.0, 5.0])
    d = np.array([2, 1, 0, 0, 0])
    C = np.array([300.0, 300.0, 300.0, 300.0, 5.0])
    sets = outcome_sets(T, d, C, "LYTIC", 30)
    flags = np.array([True, True, True, False, True])
    u = fixed_horizon_unit(flags, np.array([0.9, 0.8, 0.7, 0.1, np.nan]), sets)
    assert (u["alarms_P"], u["alarms_Q"], u["alarms_N"], u["alarms_U_excluded"]) == (1, 1, 1, 1)
    assert u["precision"] == pytest.approx(1 / 3) and u["sensitivity"] == 1.0 and u["auprc"] == 1.0
    assert u["achieved"]["alarmed_reference_rows"] == 1 and u["achieved"]["reference_rows"] == 2
    units = {(g, s): u for g in GROUPS for s in SEEDS}
    sm = fixed_horizon_summary(units)
    assert sm["precision"]["nine_folds"]["macro"] == pytest.approx(1 / 3)
    assert sm["achieved_rate_exposure_weighted"]["nine_folds"]["rate_per_cell_hour"] == pytest.approx(1.0)
    assert sm["precision_undefined_groups"] == {}


def test_the_two_cumulative_incidence_tables_are_joined_per_unit(world):
    pop, b = world
    g, s = GROUPS[3], SEEDS[2]
    pg = pop[pop["group"] == g].reset_index(drop=True)
    pr = cif_preds(b, IMAGE_SURVIVAL, g, s, pg, HORIZONS)
    for h in HORIZONS:
        assert np.array_equal(pr[h]["LYTIC"], _score(pg, h, "LYTIC", 0.01))
    extra = pd.DataFrame({"movie_id": ["x"], "track_id": ["x:1"], "t_obs_frame": [0], "group": [g]})
    assert np.isnan(cif_preds(b, IMAGE_SURVIVAL, g, s, pd.concat([pg, extra]), (30,))[30]["LYTIC"][-1])
    bad = {**b, "cumulative_incidence_30_240": b["cumulative_incidence_30_240"].iloc[1:]}
    bad.pop("hz_by_unit")
    with pytest.raises(ValueError):
        index_horizon_predictions(bad)


def test_inner_reference_scores_read_the_table_of_their_horizon(world):
    pop, b = world
    g, s = GROUPS[0], SEEDS[0]
    ref = (pop["t_obs_frame"] < 30).to_numpy()
    for h in HORIZONS:
        got = inner_ref_scores(b, g, s, pop, ref, "APOPTOSIS", h)
        want = np.concatenate([_score(pop[pop["group"] == hh].reset_index(drop=True), h, "APOPTOSIS")[
            (pop.loc[pop["group"] == hh, "t_obs_frame"] < 30).to_numpy()] for hh in inner_groups(g)])
        assert np.array_equal(got, want)


def test_horizons_block_on_a_synthetic_population(world):
    pop, b = world
    out = horizons_block(b, {h: pop for h in (30, 240)})
    assert set(out["models"]) == {DEADAHEAD, TIME}
    assert set(out["survival_estimators"]) == {DEADAHEAD, *CIF_MODELS}
    assert "not_computed" not in out
    for c in CAUSES:
        for h in ("30", "240"):
            m = out["models"][DEADAHEAD][c][h]
            assert set(m) == set(NOMINAL)
            assert set(m[PRIMARY_RATE]) >= {"precision", "sensitivity", "auprc", "prevalence", "units"}
            assert set(out["survival_estimators"][DEADAHEAD][c][h]) == {*SURV_KEYS, "ipa_groups_below_zero"}
            for rate in NOMINAL:
                for u in m[rate]["units"].values():
                    assert u["threshold"]["n_negatives"] > 0
            # the same predictions give the same AUPRC through the alarm table and through cif_auprc
            same = out["cif_models_auprc"][IMAGE_SURVIVAL][c][h]["auprc"]["nine_folds"]["macro"]
            assert np.isfinite(same) and 0 < m[PRIMARY_RATE]["auprc"]["nine_folds"]["macro"] <= 1
            assert f"{DEADAHEAD} - {DEEPHIT}|{c}|h{h}|ipa" in out["contrasts"]
    json.dumps(out, default=str)


def test_the_bundle_loader_and_parts_table_know_the_horizon_tables(tmp_path, world):
    pop, b = world
    names = ("cumulative_incidence", "cumulative_incidence_30_240", "inner_predictions", "inner_cumulative_incidence")
    rows = pop.assign(label="NO_EVENT", horizon_min=30, window_weight=1.0)
    rows.to_parquet(tmp_path / "rows.parquet", index=False)
    pd.DataFrame({"track_id": ["x"], "event_type": ["CENSORED"], "t_event": [np.nan]}).to_parquet(
        tmp_path / "events.parquet", index=False)
    pd.DataFrame({"movie_id": [f"{g}_m" for g in GROUPS], "num_frames": 160}).to_parquet(
        tmp_path / "movies.parquet", index=False)
    pop[["movie_id", "track_id", "t_obs_frame", "group"]].to_parquet(tmp_path / "landmarks.parquet", index=False)
    files = {"rows.parquet": {}, "events.parquet": {}, "movies.parquet": {}, "landmarks.parquet": {}}
    for n in names:
        b[n].to_parquet(tmp_path / f"{n}.parquet", index=False)
        files[f"{n}.parquet"] = {}
    for rel, info in files.items():
        info["sha256"] = sha256_file(tmp_path / rel)
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"files": files}))
    lb = load_bundle(tmp_path)
    assert "horizons" in PARTS and "horizons" in available_parts(lb)
    lb.pop("inner_cumulative_incidence")
    assert "horizons" not in available_parts(lb)
