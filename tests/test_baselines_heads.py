"""The LightGBM heads of the comparison methods and the DINOv2 + handcrafted survival variant, on synthetic data."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from deadahead.baselines import dinov2_survival as DS
from deadahead.baselines import heads as H
from deadahead.paths import REPO_ROOT
from deadahead.reproduce.table1 import COMPARATORS
from deadahead.survival import frames, head
from deadahead.survival.frames import GROUPS, HORIZONS_MIN

D = 4                                     # embedding width of the synthetic stores
ROUNDS = H.ROUNDS                         # the module value, before the short-fit fixture patches it


def write_store(d, movies_cells, n_frames, dim=D, seed=0, skip=()):
    """A per-frame embedding store: <movie>.npy (float16) and <movie>.parquet (cell_num, frame_index, row)."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    for m, cells in movies_cells.items():
        idx = [(c, f) for c in cells for f in range(n_frames) if (m, c, f) not in skip]
        arr = rng.normal(size=(len(idx), dim)).astype(np.float16)
        np.save(d / f"{m}.npy", arr)
        pd.DataFrame({"cell_num": [c for c, _ in idx], "frame_index": [f for _, f in idx],
                      "row": np.arange(len(idx))}).to_parquet(d / f"{m}.parquet")
    return d


def test_pool_embeddings_by_hand(tmp_path):
    store = write_store(tmp_path / "s", {"m_1": [1, 2]}, 60, skip={("m_1", 2, 25)})
    lm = pd.DataFrame({"movie_id": ["m_1", "m_1"], "track_id": ["m_1:1", "m_1:2"], "t_obs_frame": [40, 50],
                       "cell_num": [1, 2]})
    emb, cols = H.pool_embeddings(lm, H.TEMPORAL, store)
    assert emb.shape == (2, 3 * D) and emb.dtype == np.float32 and len(cols) == 3 * D
    arr = np.load(store / "m_1.npy").astype(np.float32)
    idx = pd.read_parquet(store / "m_1.parquet")
    pos = {(c, f): r for c, f, r in zip(idx["cell_num"], idx["frame_index"], idx["row"])}
    for i, (c, t) in enumerate([(1, 40), (2, 50)]):
        frames_ = [t - 5 * (8 - k) for k in range(9)]
        seq = np.stack([arr[pos[(c, f)]] if (c, f) in pos else np.zeros(D, np.float32) for f in frames_])
        assert np.allclose(emb[i, :D], seq[-1]) and np.allclose(emb[i, D:2 * D], seq.mean(0))
        assert np.allclose(emb[i, 2 * D:], seq[-1] - seq[0])
    assert (2, 25) not in pos                       # the missing frame of cell 2 pooled as zeros


def synthetic_windows(seed=0, tracks=12, n_frames=120):
    """Nine groups of one movie; 30-min rows with labels APOPTOSIS / LYTIC / UNTYPED_DEATH / NONE / censored."""
    rng = np.random.default_rng(seed)
    rows, hist = [], []
    for g in GROUPS:
        m = f"{g}_1"
        for i in range(tracks):
            t = f"{m}:{i + 1}"
            h = rng.normal(size=(n_frames, 3))
            hist.append(pd.DataFrame({"track_id": t, "frame_index": np.arange(n_frames),
                                      "f0": h[:, 0], "f1": h[:, 1], "f2": h[:, 2]}))
            for t_obs in range(40, n_frames, 10):
                lab = rng.choice(["APOPTOSIS", "LYTIC", "UNTYPED_DEATH", "NONE", "CENSORED_IN_HORIZON"],
                                 p=[0.15, 0.15, 0.1, 0.5, 0.1])
                for hz in HORIZONS_MIN:
                    rows.append({"example_id": f"{t}@{t_obs}@{hz}", "movie_id": m, "track_id": t,
                                 "t_obs_frame": t_obs, "group": g, "horizon_min": hz, "panel": "primary_439",
                                 "label": lab if hz == 30 else "NONE", "window_weight": float(rng.uniform(0.5, 1.5))})
    w = pd.DataFrame(rows)
    w = pd.concat([w, w.head(4).assign(movie_id="DU145_st_2", track_id="DU145_st_2:1")], ignore_index=True)
    return w, pd.concat(hist, ignore_index=True)


def test_forecast_rows_filters():
    w, _ = synthetic_windows()
    rows, lm = H.forecast_rows(w)
    assert (rows["panel"] == "primary_439").all() and (rows["label"] != "CENSORED_IN_HORIZON").all()
    assert "DU145_st_2" not in set(rows["movie_id"])
    assert (rows["cell_num"] == rows["track_id"].str.split(":").str[-1].astype(int)).all()
    assert not lm.duplicated(H.KEYS).any() and list(lm.columns) == [*H.KEYS, "cell_num"]


@pytest.fixture(autouse=True)
def short_heads(monkeypatch):
    monkeypatch.setattr(H, "ROUNDS", 40)                          # short fits: the tests check the plumbing
    H.set_threads(1)


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    w, hist = synthetic_windows()
    tracks = w.drop_duplicates("track_id")
    cells = {m: sorted(g["track_id"].str.split(":").str[-1].astype(int)) for m, g in tracks.groupby("movie_id")}
    store = write_store(tmp_path_factory.mktemp("store"), cells, 120)
    rows, lm = H.forecast_rows(w)
    data, dcols = H.build_table(rows, lm, hist, store)
    return w, hist, store, data, dcols


def test_build_table_and_columns(table):
    w, hist, _, data, dcols = table
    assert len(dcols) == 3 * D
    pcols = ["f0", "f1", "f2"]
    assert H.input_columns("both", dcols, pcols) == dcols + pcols
    assert H.input_columns("dino", dcols, pcols) == dcols and H.input_columns("handcrafted", dcols, pcols) == pcols
    plain, none = H.build_table(*H.forecast_rows(w), hist, None)
    assert none == [] and plain[H.KEYS + ["horizon_min"]].equals(data[H.KEYS + ["horizon_min"]])
    assert set(H.METHODS) == set(COMPARATORS)


def test_refit_leaves_out_untyped_training_rows_only(table):
    H.set_threads(1)
    _, _, _, data, dcols = table
    cols = dcols + ["f0", "f1", "f2"]
    g = "LNCaP_do"
    a, ra = H.refit_fold(data, cols, g, [1, 2], drop_untyped=False)
    b, rb = H.refit_fold(data, cols, g, [1, 2], drop_untyped=True)
    dh = data[data["horizon_min"] == 30]
    n_untyped = int(((dh["group"] != g) & (dh["label"] == "UNTYPED_DEATH")).sum())
    assert n_untyped > 0
    for c in H.CAUSES:
        assert ra[c]["removed_training_windows"]["n"] == 0
        assert rb[c]["removed_training_windows"]["n"] == n_untyped
        assert ra[c]["n_training_rows"] - rb[c]["n_training_rows"] == n_untyped
        assert ra[c]["n_rows"] == rb[c]["n_rows"] == int((dh["group"] == g).sum())
    n_te = int((dh["group"] == g).sum())
    assert len(a) == len(b) == n_te * 2 * 2                    # held-out rows x causes x seeds
    assert a[H.KEYS + ["cause", "seed"]].equals(b[H.KEYS + ["cause", "seed"]])
    assert (a["group"] == g).all() and not a.duplicated(H.KEYS + ["cause", "seed"]).any()
    again, _ = H.refit_fold(data, cols, g, [1, 2], drop_untyped=True)
    assert np.array_equal(again["score"].to_numpy(), b["score"].to_numpy())   # seeded: identical


def test_run_writes_rows_and_summary(table, tmp_path):
    H.set_threads(1)
    w, hist, store, _, _ = table
    out = H.run("dinov2_handcrafted", "PNT1A_st", ["refit", "original"], [7], store=store, windows=w, hist=hist,
                out_dir=tmp_path)
    for policy in ("refit", "original"):
        unit = H.unit_name("dinov2_handcrafted", "PNT1A_st", policy)
        rows = pd.read_parquet(tmp_path / f"{unit}.parquet")
        rec = json.loads((tmp_path / f"{unit}.json").read_text())
        assert list(rows.columns) == [*H.ROW_COLS, "cause", "seed", "score"]
        assert rec["n_features"] == 3 * D + 3 and rec["policy"] == policy and set(out[unit]) == set(H.CAUSES)
    inner = H.inner_fits(table[3], table[4] + ["f0", "f1", "f2"], "PNT1A_st", [7])
    assert set(inner["outer_group"]) == {"PNT1A_st"} and "PNT1A_st" not in set(inner["group"])
    assert set(inner["group"]) == set(GROUPS) - {"PNT1A_st"}
    with pytest.raises(FileNotFoundError):
        H.run("dinov2", "PNT1A_st", ["refit"], [7], store=tmp_path / "absent", windows=w, hist=hist, out_dir=tmp_path)


def test_run_writes_the_inner_scores(table, tmp_path):
    H.set_threads(1)
    w, hist, store, _, _ = table
    H.run("dinov2_handcrafted", "PNT1A_st", ["refit"], [7], store=store, windows=w, hist=hist, out_dir=tmp_path,
          inner=True)
    rows = pd.read_parquet(tmp_path / "dinov2_handcrafted__PNT1A_st__refit__inner.parquet")
    assert set(rows["outer_group"]) == {"PNT1A_st"} and set(rows["inner_group"]) == set(GROUPS) - {"PNT1A_st"}
    assert {"cause", "seed", "movie_id", "track_id", "t_obs_frame", "score"} <= set(rows.columns)
    assert "group" not in rows.columns


def test_auroc_ties_and_one_class():
    assert H.auroc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert H.auroc(np.array([0, 0, 1]), np.array([0.1, 0.2, 0.3])) == 1.0
    assert np.isnan(H.auroc(np.array([0, 0]), np.array([0.1, 0.2])))


def test_config_agrees_with_the_module_constants():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "baselines.yaml").read_text())
    hd = cfg["heads"]
    assert hd["lightgbm"] == {k: v for k, v in H.LGB.items() if k != "num_threads"}
    assert hd["rounds"] == ROUNDS == 300 and hd["horizon_min"] == H.HORIZON
    assert hd["pooling"] == H.TEMPORAL and set(hd["methods"]) == set(H.METHODS)
    assert hd["policies"] == list(H.POLICIES)
    assert cfg["dinov2_survival"]["width"] == DS.WIDTH


# ------------------------------------------------------------------ DINOv2 + handcrafted through the survival head
def test_dinov2_survival_end_to_end(tmp_path, monkeypatch):
    from tests.test_survival_head import synthetic_tables, write_inputs
    monkeypatch.setattr(head, "ROUNDS", 20)                       # a short fit: the test checks the plumbing
    tables, hist = synthetic_tables(tracks_per_group=6)
    d = write_inputs(tmp_path, tables, hist)
    tracks = tables["windows"].drop_duplicates("track_id")
    cells = {m: sorted(g["track_id"].str.split(":").str[-1].astype(int)) for m, g in tracks.groupby("movie_id")}
    store = write_store(tmp_path / "store", cells, 160, dim=DS.DINO_DIM)
    args = ["--group", "LNCaP_do", "--seed", "2601562", "--tables-dir", str(d), "--history",
            str(tmp_path / "history.parquet"), "--store", str(store), "--threads", "2", "--out-dir", str(tmp_path / "o")]
    assert DS.main(args) == 0
    name = f"{DS.METHOD}__LNCaP_do__s2601562__primary"
    p = np.load(tmp_path / "o" / "pred" / f"{name}.npz")
    held = tables["windows"][tables["windows"]["group"] == "LNCaP_do"]
    assert set(p["index_track"]) <= set(held["track_id"]) and int(p["n_nonfinite"]) == 0
    for h in HORIZONS_MIN:
        s = p["p_APOPTOSIS_" + str(h)] + p["p_LYTIC_" + str(h)]
        assert ((s >= 0) & (s <= 1.0 + 1e-12)).all()
    booster = head.lgb.Booster(model_file=str(tmp_path / "o" / "boosters" / f"{name}.txt"))
    assert booster.num_feature() == DS.WIDTH
    assert frames.N_HANDCRAFTED + DS.BLOCK_I + 1 == DS.WIDTH == 2417


def test_landmark_list_mapping():
    f = pd.DataFrame({"movie_id": ["a", "a", "a"], "track_id": ["a:1", "a:1", "a:2"], "t_obs_frame": [40, 40, 40]})
    lm = DS.landmark_list(f)
    assert list(lm["cell_num"]) == [1, 2]
    with pytest.raises(ValueError):
        DS.landmark_list(pd.DataFrame({"movie_id": ["a"], "track_id": ["a:x"], "t_obs_frame": [40]}))


def test_heads_read_the_stores_the_embedding_step_writes():
    pytest.importorskip("torch")
    from deadahead.baselines.dinov2_common import embedding_store
    for g in GROUPS:
        assert H.default_store("dinov2", g) == embedding_store("adapted", g)
        assert H.default_store("dinov2_finetuned", g) == embedding_store("finetuned", g)
