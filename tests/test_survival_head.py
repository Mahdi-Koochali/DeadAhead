"""The survival head, its labels, its saved boosters and one held-out fit on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from deadahead.features import frame_measures
from deadahead.features.history import HISTORY_COLUMNS
from deadahead.survival import frames, head, train
from deadahead.survival.boosters import (
    base_matrix,
    capture_booster,
    load_booster,
    max_abs_diff,
    predict_incidence,
    save_booster,
)
from deadahead.survival.frames import (
    FEATURE_DIM,
    GROUPS,
    HORIZONS_MIN,
    WIDTH,
    apply_view,
    coherence_violations,
    input_columns,
    panel_view,
    prediction_payload,
)
from deadahead.survival.config import head_config, windows_config
from deadahead.survival.oracle import build_oracle

CFG = {"survival_grid": {"bin_frames": 10, "n_bins": 8}}
CFG_W = {"frame_interval_min": 3.0, "horizons_min": [30, 60, 120, 240]}
COLS = ["x0", "x1", "x2"]


def synthetic_landmarks(n=1500, seed=0):
    """Landmarks whose hazard of each cause rises with one feature; some are censored."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    d = pd.DataFrame(X, columns=COLS)
    d["movie_id"] = np.where(np.arange(n) % 2 == 0, "m_1", "m_2")
    d["track_id"] = [f"{m}:{i}" for i, m in enumerate(d["movie_id"])]
    d["t_obs_frame"] = 40
    p_apo = 1 / (1 + np.exp(-(X[:, 0] - 1.0)))
    p_lyt = 1 / (1 + np.exp(-(X[:, 1] - 1.0)))
    u = rng.random(n)
    cause = np.where(u < 0.5 * p_apo, "APOPTOSIS", np.where(u < 0.5 * (p_apo + p_lyt), "LYTIC", "NONE"))
    dt = rng.integers(1, 90, size=n)
    d["true_event"] = cause
    d["t_event_true"] = np.where(cause != "NONE", 40 + dt, np.nan)
    cens = (cause == "NONE") & (rng.random(n) < 0.5)
    d["censor_offset"] = np.where(cens, 10.0 * rng.integers(0, 8, size=n), np.inf)
    d["window_weight"] = 1.0
    return d


@pytest.fixture(scope="module")
def fitted():
    head.set_threads(1)
    tr, te = synthetic_landmarks(1500, 0), synthetic_landmarks(400, 1)
    (out, boosters) = capture_booster(head.train_predict_survival, tr, te, COLS, CFG, CFG_W, 7)
    return tr, te, out, boosters


def test_incidence_non_decreasing_and_bounded(fitted):
    _, te, out, _ = fitted
    assert sorted(k for k in out if k != "_index") == list(HORIZONS_MIN)
    for h0, h1 in zip(HORIZONS_MIN[:-1], HORIZONS_MIN[1:]):
        for c in head.CAUSES:
            assert (out[h1][c] >= out[h0][c]).all()
    for h in HORIZONS_MIN:
        a, b = out[h]["APOPTOSIS"], out[h]["LYTIC"]
        assert (a >= 0).all() and (b >= 0).all()
        assert (a + b <= 1.0 + 1e-12).all()
    assert coherence_violations(out) == 0
    assert len(out["_index"]) == len(te)


def test_seeded_determinism(fitted):
    tr, te, out, _ = fitted
    again = head.train_predict_survival(tr, te, COLS, CFG, CFG_W, 7)
    other = head.train_predict_survival(tr, te, COLS, CFG, CFG_W, 8)
    for h in HORIZONS_MIN:
        for c in head.CAUSES:
            assert np.array_equal(out[h][c], again[h][c])
    assert max_abs_diff(out, other) > 0


def test_saved_booster_reproduces_the_head(fitted, tmp_path):
    _, te, out, boosters = fitted
    assert len(boosters) == 1
    keys, Xt = base_matrix(te, COLS)
    assert keys.equals(out["_index"])
    assert max_abs_diff(predict_incidence(boosters[0], Xt, CFG, CFG_W), out) == 0.0
    p = save_booster(boosters[0], tmp_path / "head.txt")
    assert max_abs_diff(predict_incidence(load_booster(p), Xt, CFG, CFG_W), out) <= 1e-12


def test_one_class_returns_nothing():
    tr = synthetic_landmarks(300, 2).assign(true_event="NONE", t_event_true=np.nan)
    assert head.train_predict_survival(tr, tr, COLS, CFG, CFG_W, 7) == {}


def test_oracle_labels():
    windows = pd.DataFrame({"movie_id": ["m_1", "m_1"], "track_id": ["m_1:1", "m_1:2"], "t_obs_frame": [10, 10],
                            "group": "g", "cell_line": "c", "treatment": "t", "window_weight": 1.0})
    events = pd.DataFrame({"track_id": ["m_1:1", "m_1:2"], "event_type": ["LYTIC", "NONE"],
                           "t_event": [25.0, np.nan]})
    movies = pd.DataFrame({"movie_id": ["m_1"], "num_frames": [35]})
    o = build_oracle(windows, events, movies, CFG_W).set_index(["track_id", "horizon_min"])
    # the death at frame 25 is after the 30-min horizon (frame 20) and inside the 60-min one (frame 30)
    assert o.loc[("m_1:1", 30), "oracle_label"] == 0 and o.loc[("m_1:1", 60), "oracle_label"] == 2
    assert o.loc[("m_1:1", 240), "oracle_label"] == 2
    # the survivor is known event-free to frame 34: determined at 30 and 60 min, not at 120 or 240 min
    assert o.loc[("m_1:2", 30), "oracle_label"] == 0 and o.loc[("m_1:2", 60), "oracle_label"] == 0
    assert ("m_1:2", 120) not in o.index and ("m_1:2", 240) not in o.index


def test_panel_view_training_labels():
    lm = pd.DataFrame({"movie_id": ["m_1"] * 3, "track_id": ["m_1:1", "m_1:2", "m_1:3"], "t_obs_frame": [40] * 3,
                       "time_to_event_min": [45.0, 45.0, 30.0],
                       "track_terminal_event": ["LYTIC", "CENSORED", "UNTYPED_DEATH"]})
    v = panel_view(lm).set_index("track_id")
    assert v.loc["m_1:1", "true_event"] == "LYTIC" and v.loc["m_1:1", "t_event_true"] == 55
    assert np.isinf(v.loc["m_1:1", "censor_offset"])
    # 15 frames of follow-up: one full bin survived, the partial second bin is not credited
    assert v.loc["m_1:2", "true_event"] == "NONE" and v.loc["m_1:2", "censor_offset"] == 10
    assert v.loc["m_1:3", "true_event"] == "NONE" and v.loc["m_1:3", "censor_offset"] == 0
    tr = lm[["movie_id", "track_id", "t_obs_frame"]].assign(true_event="X", horizon_min=30)
    out = apply_view(pd.concat([tr, tr.assign(horizon_min=60)], ignore_index=True), v.reset_index())
    assert list(out["true_event"]) == ["LYTIC", "NONE", "NONE"] * 2


def test_input_columns_and_payload(fitted):
    video, hand = [f"v{i:04d}" for i in range(1024)], [f"h{i}" for i in range(112)]
    assert len(input_columns("deadahead", video, hand)) + 1 == WIDTH["deadahead"] == 1137
    assert input_columns("handcrafted_survival", video, hand) == hand
    _, te, out, _ = fitted
    te = pd.concat([te.assign(horizon_min=h, oracle_label=0) for h in HORIZONS_MIN], ignore_index=True)
    pay = prediction_payload(out, te)
    assert int(pay["n_nonfinite"]) == 0
    for h in HORIZONS_MIN:
        assert pay[f"so_mask_{h}"].all() and (pay[f"so_y_{h}"] == 0).all()


# ------------------------------------------------------------------ one held-out fit end to end
N_FRAMES = 160


def synthetic_tables(tracks_per_group=14, seed=0):
    """Nine groups of one movie each; tracks from frame 0, landmarks every 10 frames from frame 40."""
    rng = np.random.default_rng(seed)
    win, ev, mov, hist = [], [], [], []
    for g in GROUPS:
        m = f"{g}_1"
        mov.append({"movie_id": m, "num_frames": N_FRAMES})
        for i in range(tracks_per_group):
            t = f"{m}:{i + 1}"
            kind = rng.choice(["APOPTOSIS", "LYTIC", "NONE"], p=[0.3, 0.3, 0.4])
            t_ev = int(rng.integers(55, N_FRAMES)) if kind != "NONE" else None
            ev.append({"track_id": t, "event_type": kind, "t_event": float(t_ev) if t_ev else np.nan})
            end = t_ev if t_ev else N_FRAMES - 1
            h = pd.DataFrame(rng.normal(size=(N_FRAMES, len(HISTORY_COLUMNS))), columns=HISTORY_COLUMNS)
            if t_ev:                                   # the measures drift before the death
                h.iloc[:, 0] += np.clip(np.arange(N_FRAMES) - (t_ev - 30), 0, None) / 10.0
            hist.append(pd.concat([pd.DataFrame({"track_id": t, "frame_index": np.arange(N_FRAMES)}), h], axis=1))
            for t_obs in range(40, end, 10):
                for hz in HORIZONS_MIN:
                    win.append({"movie_id": m, "track_id": t, "t_obs_frame": t_obs, "group": g,
                                "cell_line": g.split("_")[0], "treatment": g.split("_")[1], "window_weight": 1.0,
                                "panel": "primary_439", "horizon_min": hz, "time_to_event_min": 3.0 * (end - t_obs),
                                "track_terminal_event": kind if t_ev else "CENSORED"})
    tables = {"windows": pd.DataFrame(win), "events": pd.DataFrame(ev), "movies": pd.DataFrame(mov)}
    return tables, pd.concat(hist, ignore_index=True)


def write_inputs(tmp_path, tables, hist, seed=0):
    d = tmp_path / "tables"
    d.mkdir()
    for n, t in tables.items():
        t.to_parquet(d / f"{n}.parquet")
    hist.to_parquet(tmp_path / "history.parquet")
    rows = tables["windows"].drop_duplicates(subset=["movie_id", "track_id", "t_obs_frame"])[
        ["movie_id", "track_id", "t_obs_frame", "group"]].reset_index(drop=True)
    rows.to_parquet(tmp_path / "rows.parquet")
    feats = np.random.default_rng(seed).normal(size=(len(rows), FEATURE_DIM)).astype(np.float32)
    np.save(tmp_path / "feats.npy", feats)
    return d


@pytest.mark.parametrize("method", ["handcrafted_survival", "deadahead"])
def test_train_then_predict_from_saved_booster(tmp_path, method):
    tables, hist = synthetic_tables()
    d = write_inputs(tmp_path, tables, hist)
    common = ["--group", "LNCaP_do", "--seed", "2601562", "--tables-dir", str(d), "--history",
              str(tmp_path / "history.parquet"), "--method", method, "--threads", "2"]
    if method == "deadahead":
        common += ["--features", str(tmp_path / "feats.npy"), "--rows", str(tmp_path / "rows.parquet")]
    assert train.main(common + ["--out-dir", str(tmp_path / "fit")]) == 0
    name = f"{method}__LNCaP_do__s2601562__primary"
    booster = tmp_path / "fit" / "boosters" / train.booster_name(method, "LNCaP_do", 2601562, "primary")
    if method == "deadahead":
        assert booster.name == "deadahead_head__LNCaP_do__seed2601562.txt"
    assert train.main(common + ["--out-dir", str(tmp_path / "pred"), "--booster", str(booster)]) == 0
    a = np.load(tmp_path / "fit" / "pred" / f"{name}.npz")
    b = np.load(tmp_path / "pred" / "pred" / f"{name}.npz")
    assert sorted(a.files) == sorted(b.files)
    for k in a.files:
        assert np.array_equal(a[k], b[k]), k
    held = tables["windows"][tables["windows"]["group"] == "LNCaP_do"]
    assert set(a["index_track"]) <= set(held["track_id"])
    for h in HORIZONS_MIN:
        s = a["p_APOPTOSIS_" + str(h)] + a["p_LYTIC_" + str(h)]
        assert ((s >= 0) & (s <= 1.0 + 1e-12)).all()


def test_fold_is_the_group_option_and_the_adapted_features_are_required(tmp_path, capsys):
    tables, hist = synthetic_tables()
    d = write_inputs(tmp_path, tables, hist)
    common = ["--fold", "LNCaP_do", "--seed", "2601562", "--tables-dir", str(d), "--history",
              str(tmp_path / "history.parquet"), "--threads", "2", "--out-dir", str(tmp_path / "fit")]
    with pytest.raises(SystemExit) as e:
        train.main(common + ["--method", "videomae_adapted_survival", "--rows", str(tmp_path / "rows.parquet")])
    assert e.value.code == 2 and "needs --features" in capsys.readouterr().err
    assert train.main(common + ["--method", "handcrafted_survival"]) == 0
    assert (tmp_path / "fit" / "boosters" / "handcrafted_survival__LNCaP_do__s2601562__primary.txt").exists()


def inner_args(tmp_path, tables, hist, group="LNCaP_do", inner="PNT1A_do"):
    d = write_inputs(tmp_path, tables, hist)
    return ["--group", group, "--seed", "2601562", "--inner-group", inner, "--tables-dir", str(d), "--history",
            str(tmp_path / "history.parquet"), "--features", str(tmp_path / "feats.npy"), "--rows",
            str(tmp_path / "rows.parquet"), "--threads", "2"]


def test_inner_group_fits_seven_groups_and_predicts_the_inner_group(tmp_path, monkeypatch):
    monkeypatch.setattr(head, "ROUNDS", 20)                       # a short fit: the test checks the plumbing
    tables, hist = synthetic_tables()
    common = inner_args(tmp_path, tables, hist)
    seen = {}
    fit = head.train_predict_survival

    def recording_fit(tr, te, *args, **kwargs):
        seen.update(train=set(tr["group"]), predicted=set(te["group"]), censored=te["censor_offset"].to_numpy())
        return fit(tr, te, *args, **kwargs)
    monkeypatch.setattr(head, "train_predict_survival", recording_fit)
    assert train.main(common + ["--out-dir", str(tmp_path / "fit")]) == 0
    assert seen["train"] == set(GROUPS) - {"LNCaP_do", "PNT1A_do"} and seen["predicted"] == {"PNT1A_do"}
    assert np.isinf(seen["censored"]).all()
    name = "deadahead__LNCaP_do__s2601562__primary__hPNT1A_do"
    a = np.load(tmp_path / "fit" / "pred" / f"{name}.npz")
    w = tables["windows"]
    assert set(a["index_track"]) == set(w.loc[w["group"] == "PNT1A_do", "track_id"])
    booster = tmp_path / "fit" / "boosters" / "deadahead_head__LNCaP_do__seed2601562__hPNT1A_do.txt"
    assert train.main(common + ["--out-dir", str(tmp_path / "pred"), "--booster", str(booster)]) == 0
    b = np.load(tmp_path / "pred" / "pred" / f"{name}.npz")
    for k in a.files:
        assert np.array_equal(a[k], b[k]), k


def test_inner_head_differs_from_the_outer_head_of_the_same_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(head, "ROUNDS", 20)
    tables, hist = synthetic_tables()
    common = inner_args(tmp_path, tables, hist)
    assert train.main(common + ["--out-dir", str(tmp_path / "o")]) == 0
    outer = [c for i, c in enumerate(common) if c != "--inner-group" and common[i - 1] != "--inner-group"]
    outer[outer.index("LNCaP_do")] = "PNT1A_do"
    assert train.main(outer + ["--out-dir", str(tmp_path / "o")]) == 0
    a = np.load(tmp_path / "o" / "pred" / "deadahead__LNCaP_do__s2601562__primary__hPNT1A_do.npz")
    b = np.load(tmp_path / "o" / "pred" / "deadahead__PNT1A_do__s2601562__primary.npz")
    assert a["index_track"].tolist() == b["index_track"].tolist()
    assert not np.array_equal(a["p_LYTIC_30"], b["p_LYTIC_30"])


def test_inner_group_refuses_the_held_out_group(tmp_path, capsys):
    tables, hist = synthetic_tables()
    with pytest.raises(SystemExit) as e:
        train.main(inner_args(tmp_path, tables, hist, inner="LNCaP_do") + ["--out-dir", str(tmp_path / "o")])
    assert e.value.code == 2 and "--inner-group" in capsys.readouterr().err
    tr = pd.DataFrame({"group": ["DU145_bp"]})
    with pytest.raises(ValueError):
        frames.inner_split(tr, tr, "DU145_do", "DU145_do")
    with pytest.raises(AssertionError):
        frames.inner_split(tr.assign(group="DU145_do"), tr, "DU145_do", "DU145_bp")
    with pytest.raises(AssertionError):                 # six training groups: one is missing
        frames.inner_split(tr, tr, "DU145_do", "DU145_st")


def test_untyped_option_has_its_own_files(tmp_path, monkeypatch):
    monkeypatch.setattr(head, "ROUNDS", 20)
    tables, hist = synthetic_tables()
    d = write_inputs(tmp_path, tables, hist)
    common = ["--group", "LNCaP_do", "--seed", "2601562", "--method", "handcrafted_survival", "--tables-dir", str(d),
              "--history", str(tmp_path / "history.parquet"), "--threads", "2", "--out-dir", str(tmp_path / "o")]
    assert train.main(common) == 0 and train.main(common + ["--untyped", "at_risk_to_cut"]) == 0
    stem = "handcrafted_survival__LNCaP_do__s2601562__primary"
    assert sorted(p.name for p in (tmp_path / "o" / "pred").iterdir()) == [f"{stem}.npz", f"{stem}__at_risk_to_cut.npz"]
    assert sorted(p.name for p in (tmp_path / "o" / "boosters").iterdir()) == [f"{stem}.txt",
                                                                               f"{stem}__at_risk_to_cut.txt"]


def test_no_training_landmark_is_at_risk_past_the_cut():
    lm = pd.DataFrame({"movie_id": ["m_1"] * 3, "track_id": ["m_1:1", "m_1:2", "m_1:3"], "t_obs_frame": [400] * 3,
                       "time_to_event_min": [45.0, 114.0, 30.0],
                       "track_terminal_event": ["LYTIC", "CENSORED", "UNTYPED_DEATH"]})
    movies = pd.DataFrame({"movie_id": ["m_1"], "num_frames": [500]})
    v = panel_view(lm)
    assert frames.past_cut_count(v, movies, frames.CUT_FRAME) == 0
    assert frames.past_cut_count(v.assign(censor_offset=np.inf), movies, frames.CUT_FRAME) == 2
    with pytest.raises(AssertionError):
        frames.assert_inside_cut(v.assign(censor_offset=np.inf), movies)


def test_at_risk_to_cut_follows_an_untyped_death_to_the_cut():
    lm = pd.DataFrame({"movie_id": ["m_1"], "track_id": ["m_1:1"], "t_obs_frame": [400], "time_to_event_min": [30.0],
                       "track_terminal_event": ["UNTYPED_DEATH"]})
    movies = pd.DataFrame({"movie_id": ["m_1"], "num_frames": [500]})
    tte, term = frames.reference_inputs(lm, "at_risk_to_cut", movies, frames.CUT_FRAME)
    assert tte[0] == 3.0 * (frames.CUT_FRAME - 400) and term[0] == "CENSORED"
    tte, term = frames.reference_inputs(lm, "censor_at_death")
    assert tte[0] == 30.0 and term[0] == "UNTYPED_DEATH"
    with pytest.raises(ValueError):
        frames.reference_inputs(lm, "at_risk_to_cut")


def test_panels_differ_by_the_excluded_movie_only():
    w = pd.DataFrame({"panel": [frames.PANEL_NAME] * 2 + ["full_length"],
                      "movie_id": [frames.EXCLUDED_MOVIE, "DU145_st_1", "DU145_st_1"]})
    assert list(frames.panel_windows(w, "primary")["movie_id"]) == ["DU145_st_1"]
    assert list(frames.panel_windows(w, "all_movies")["movie_id"]) == [frames.EXCLUDED_MOVIE, "DU145_st_1"]


def test_config_files_agree_with_the_module_constants():
    cfg, cfg_w = head_config(), windows_config()
    assert (cfg["survival_grid"]["bin_frames"], cfg["survival_grid"]["n_bins"]) == (frames.BIN_FRAMES, frames.N_BINS)
    assert cfg["protocol"]["panel"] == frames.PANEL_NAME
    assert cfg["protocol"]["morphology_excluded_movies"] == [frames.EXCLUDED_MOVIE]
    assert set(cfg["protocol"]["morphology_excluded_movies"]) == frame_measures.MORPHOLOGY_EXCLUDED
    assert tuple(cfg["protocol"]["seeds"]) == frames.SEEDS
    assert float(cfg_w["frame_interval_min"]) == frames.FRAME_MIN
    assert tuple(cfg_w["horizons_min"]) == frames.HORIZONS_MIN
    assert int(cfg_w["truncation"]["primary_truncation_frame"]) - 1 == frames.CUT_FRAME
    assert int(cfg_w["landmark"]["grid_every_frames"]) == frames.BIN_FRAMES
