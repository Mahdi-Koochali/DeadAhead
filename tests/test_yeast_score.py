"""Yeast probe on synthetic data: lifespans, rows, clips, picture tables, the rules, the reading, lead times, survival
scores and the scorer. The numbers of S7 are checked on the released bundle when it is installed."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from deadahead.paths import REPO_ROOT
from deadahead.yeast import constants as C
from deadahead.yeast import data as D
from deadahead.yeast import lifespans as L
from deadahead.yeast import rules as B
from deadahead.yeast import score as S

pytest.importorskip("lightgbm")


def synthetic_series(n_traps: int = 5, n_frames: int = 240, seed: int = 0) -> dict:
    """{trap: class array} over the 7 positions: budding cycles, some deaths, some clogs, one empty trap."""
    rng = np.random.default_rng(seed)
    out = {}
    for p in C.POSITIONS:
        for i in range(n_traps):
            c = np.zeros(n_frames, dtype=int) + L.UNBUD
            start = int(rng.integers(0, 10))
            c[:start] = L.EMPTY
            t, cyc = start, int(rng.integers(12, 18))
            while t < n_frames:
                c[t:t + cyc // 2] = L.UNBUD
                c[t + cyc // 2:t + cyc - 3] = L.SMALL
                c[t + cyc - 3:t + cyc] = L.LARGE
                t += cyc
                cyc += int(rng.integers(0, 3))
            if i % 4:                                        # dead, or clogged for every fourth trap
                end = int(rng.integers(n_frames // 2, n_frames - 10))
                c[end:] = L.CLOG if i % 4 == 3 else L.DEAD
            out[f"{p}_{i + 1}"] = c
    out[f"{C.POSITIONS[0]}_99"] = np.full(n_frames, L.EMPTY)
    return out


def test_lifespan_and_buds():
    c = np.array([5, 1, 1, 2, 3, 3, 1, 2, 3, 2, 4, 4])
    assert L.detecdiv_lifespan(c) == (1, 10, True)
    assert L.detecdiv_buds(c, 1, 10).tolist() == [6, 9]
    assert L.detecdiv_lifespan(np.array([1, 2, 3, 1])) == (0, 4, False)
    assert L.detecdiv_lifespan(np.array([5, 5])) is None
    assert L.window_targets(0, 6, 5, 6) == 1 and L.window_targets(0, 7, 6, 6) == 0
    assert L.window_targets(0, None, 3, 6) is None and L.window_targets(0, None, 6, 6) == 0


def test_rows_and_row_files(tmp_path):
    series = synthetic_series()
    rows, notes = D.build_rows(series)
    assert notes["n_traps_no_cell"] == 1 and notes["n_lifespans"] == 7 * 5
    assert set(rows["position"]) == set(C.POSITIONS)
    starts = D.lifespan_table(series).set_index("unit")["start"]
    for unit, g in rows.groupby("unit"):
        assert g["t"].iloc[0] == starts[unit] and (np.diff(g["t"]) == C.LANDMARK_STEP).all()
    ev = rows["ev_off"].to_numpy()
    assert np.all(np.isnan(ev) | (ev > 0))
    rec = D.write_row_files(rows, tmp_path)
    for p in C.POSITIONS:
        f = D.read_row_file(tmp_path, "final", p)
        assert p not in set(f["position"]) and len(set(f["position"])) == 6
        assert len(D.label_efficiency_positions(p)) == 3
        assert rec["folds"][p]["threshold_position"] in C.training_positions(p)


def test_clips_are_past_only():
    for t in (0, 5, 24, 300):
        f = D.clip_frames(t)
        assert f[-1] == t and f.max() == t and (np.diff(f) >= 0).all() and len(f) == 16
    x = np.array([0, 128, 255], dtype=np.uint8)
    y = D.to_unit(x, 0.5, 0.1, 0.3, 0.15)
    assert y.dtype == np.float32 and y.min() >= 0 and y.max() <= 1


def fake_store(tmp_path, series, seed: int = 1):
    rng = np.random.default_rng(seed)
    d = tmp_path / "vid_store"
    d.mkdir()
    for trap, c in series.items():
        np.save(d / f"{trap}.npy", rng.integers(0, 256, size=(len(c), *C.CROP_SHAPE), dtype=np.uint8))
    return D.VidStore(d)


def test_picture_tables_round_trip(tmp_path):
    series = synthetic_series(n_traps=2, n_frames=80)
    rows, _ = D.build_rows(series)
    store = fake_store(tmp_path, series)
    folds = {p: {"channel": i % 2, "m_D": 0.3, "s_D": 0.1, "size_rule": {"threshold": 100 + i, "side": "below"}}
             for i, p in enumerate(C.POSITIONS)}
    bright, size = D.picture_tables(rows, store, folds)
    pics = D.fold_pictures(bright, size, folds, rows)
    keys = rows[["unit", "position", "t"]]
    for i, p in enumerate(C.POSITIONS):
        f = folds[p]
        ref = D.picture_features(keys, store, f["channel"], f["size_rule"]["threshold"], "below")
        got = pics[p]
        assert got.index.equals(rows.index)
        for c in D.PIC_SIZE:
            assert np.array_equal(got[c].to_numpy(), ref[c].to_numpy())


def write_bundle(tmp_path, series, n_seeds: int = 3) -> tuple:
    labels = tmp_path / "bundle" / "labels"
    labels.mkdir(parents=True)
    store = fake_store(tmp_path, series)
    folds = {p: {"channel": 0, "m_D": 0.3, "s_D": 0.1, "size_rule": {"threshold": 120, "side": "below"}}
             for p in C.POSITIONS}
    (labels / "folds.json").write_text(json.dumps({"folds": folds, "source_moments": {"m_F": 0.27, "s_F": 0.15}}))
    rows = D.write_tables(series, labels, store, folds)
    pred = tmp_path / "bundle" / "predictions"
    pred.mkdir()
    rng = np.random.default_rng(3)
    for st in C.STARTS:
        for p in C.POSITIONS:
            g = rows[rows["position"] == p][["unit", "position", "t"]].reset_index(drop=True)
            for s in C.SEEDS[:n_seeds]:
                y = g.merge(rows[["unit", "t", "y_30"]], on=["unit", "t"])["y_30"].fillna(0).to_numpy()
                p_death = np.clip(0.2 * y + rng.random(len(g)) * 0.5, 0, 1)
                g.assign(p_none=1 - p_death, p_death=p_death).to_parquet(pred / f"{S.run_name(st, p, s)}.parquet")
    return tmp_path / "bundle", rows


def add_survival(bundle, rows, n_seeds: int = 3) -> None:
    """A survival/ folder: the S, SL and NB frames of every final run and the S frame of every threshold run."""
    surv = bundle / "survival"
    surv.mkdir()
    rng = np.random.default_rng(4)

    def cif(keys, stem):
        base = rng.random(len(keys))
        return keys.assign(**{f"{stem}{h}": base * h / 240 for h in C.HORIZONS})

    def keys_of(p):
        return rows[rows["position"] == p][["unit", "position", "t"]].reset_index(drop=True)

    for st in C.STARTS:
        for p in C.POSITIONS:
            for s in C.SEEDS[:n_seeds]:
                u = S.run_name(st, p, s)
                cif(keys_of(p), "cif_").to_parquet(surv / f"{u}__S.parquet")
                cif(keys_of(p), "cif_").to_parquet(surv / f"{u}__SL.parquet")
                cif(cif(keys_of(p), "cif_bud_"), "cif_death_").to_parquet(surv / f"{u}__NB.parquet")
            cif(keys_of(C.threshold_position(p)), "cif_").to_parquet(surv / f"thr__{st}__{p}__S.parquet")


def test_reading_classes():
    d = {p: 0.05 for p in C.POSITIONS}
    r = S.classify(d)
    assert r["class"] == S.CLASS_NAMES["bar"][0] and r["k_plus"] == 7
    assert r["sign_flip_p"] == pytest.approx(2 / 2 ** 7)
    d[C.POSITIONS[0]] = -0.01
    d[C.POSITIONS[1]] = -0.01
    r = S.classify(d)
    assert r["k_plus"] == 5 and r["class"] == S.CLASS_NAMES["bar"][3]
    r = S.classify({p: 0.001 * (i - 3) for i, p in enumerate(C.POSITIONS)}, "transfer")
    assert r["class"] == "NO TANGIBLE DIFFERENCE" and r["mean_delta"] == pytest.approx(0.0)


def test_rules_one_position(tmp_path, fast_rules):
    series = synthetic_series(n_traps=3, n_frames=200)
    rows, _ = D.build_rows(series)
    store = fake_store(tmp_path, series)
    folds = {p: {"channel": 0, "m_D": 0.3, "s_D": 0.1, "size_rule": {"threshold": 120, "side": "below"}}
             for p in C.POSITIONS}
    bright, size = D.picture_tables(rows, store, folds)
    pics = D.fold_pictures(bright, size, folds, rows)
    p = C.POSITIONS[0]
    res = B.bar_for_position(rows, pics[p], p, horizons=(30,))[30]
    fits = {k: v["auprc"] for k, v in res["members"].items() if "auprc" in v}
    assert res["bar"]["auprc"] == max(fits.values())
    assert res["n_landmarks"] == int(rows[(rows["position"] == p) & rows["y_30"].notna()].shape[0])
    assert 0 < res["event_rate"] < 1


def test_score_bundle_end_to_end(tmp_path, fast_rules):
    bundle, rows = write_bundle(tmp_path, synthetic_series(n_traps=3, n_frames=200))
    rec = S.score_bundle(bundle, with_sensitivity=False, verify=False)
    s = rec["summary"]
    for st in C.STARTS:
        assert rec["scorable"][st]["scorable"]
        assert s["macro_auprc_30min"][st] == pytest.approx(np.mean([rec["a_R"][st][p] for p in C.POSITIONS]))
    deltas = {p: rec["a_R"][C.STARTS[0]][p] - rec["bar"][p]["auprc"] for p in C.POSITIONS}
    assert s["ahead_of_best_rule"][C.STARTS[0]] == sum(v > 0 for v in deltas.values())
    assert rec["comparison_B"]["k_plus"] == sum(rec["a_R"][C.STARTS[0]][p] > rec["a_R"][C.STARTS[1]][p]
                                                for p in C.POSITIONS)
    assert rec["intervals"][C.POSITIONS[0]]["n_resamples"] == 50
    assert set(rec["bar_lead_time"]) == set(C.POSITIONS)
    assert "survival_scores" not in rec                       # no survival/ folder in this bundle
    total = sum(rec["bar"][p]["n_landmarks"] for p in C.POSITIONS)
    positives = sum(rec["bar"][p]["n_positive"] for p in C.POSITIONS)
    for split in ("phase", "since_last_bud"):                 # each split partitions the scored landmarks
        groups = rec["phase_split"][split].values()
        assert sum(g["n_landmarks"] for g in groups) == total and sum(g["n_positive"] for g in groups) == positives
        assert all(set(g["pooled"]) == {"bar", *C.STARTS} for g in groups)


def test_score_bundle_reads_the_survival_layer_and_the_sensitivity(tmp_path, fast_rules):
    series = synthetic_series(n_traps=4, n_frames=200)
    bundle, rows = write_bundle(tmp_path, series)
    add_survival(bundle, rows)
    rec = S.score_bundle(bundle, verify=False)
    st = C.STARTS[0]
    per = rec["survival_scores"]["per_position"][C.POSITIONS[0]]
    assert per["survival_layer"][st]["mean_over_seeds"]["n_averaged"] == 3
    assert set(per["rules"]) >= {"label_surv"}
    macro = rec["survival_scores"]["macro_over_positions"][f"survival_layer:{st}"]
    assert macro["n_positions"] == 7 and 0 <= macro["c_index_of_the_4h_risk"] <= 1
    cell = rec["next_bud_vs_death"][st][C.POSITIONS[0]]
    assert set(cell) == {"30", "60", "120", "240"} and set(cell["240"]) == {"bud", "death"}
    lt = rec["lead_time"]["per_start"][st]
    assert lt[C.POSITIONS[0]]["threshold_position"] == C.threshold_position(C.POSITIONS[0])
    assert 0 <= lt["k_of_7_longer_than_bar"] <= 7
    assert set(rec["secondary_horizons"][st]) == {"30", "60", "120", "240"}
    # the empty trap has no lifespan; turning the lifespans that end clogged into removed ones drops them
    ends = D.lifespan_table(series)
    sens = rec["sensitivity"]
    assert sens["empty_as_death"]["n_deaths"] == int(ends["event"].sum())
    assert sens["clog_empty_left_out"]["n_lifespans"] == len(ends) - 7          # one clogged trap per position
    assert set(sens["empty_as_death"]["readings"]) == set(C.STARTS)


def test_alarm_onset_by_hand():
    from deadahead.yeast import leadtime as LT
    assert LT.alarm_onset([0.1, 0.9, 0.9, 0.1, 0.9], 0.5) == 1                  # one landmark below is allowed
    assert LT.alarm_onset([0.1, 0.9, 0.9, 0.1, 0.9], 0.5, allow_below=0) == 4
    assert LT.alarm_onset([0.9, 0.1, 0.1, 0.9], 0.5) == 3                       # two below end the alarm
    assert LT.alarm_onset([0.1, 0.5], 0.5) is None                              # at the threshold is not above it


def test_lead_times_by_hand():
    from deadahead.yeast import leadtime as LT
    flagged = {"t": [0, 6, 12, 18], "s": [0.1, 0.9, 0.1, 0.9], "event_frame": 24, "buds": [7, 13, 30],
               "neg": [True, False, False, False]}                              # onset at frame 6: 18 frames, 90 min
    never = {"t": [0, 6], "s": [0.1, 0.2], "event_frame": 12, "buds": [], "neg": [True, False]}
    capped = {"t": [0, 6], "s": [0.9, 0.9], "event_frame": 200, "buds": [], "neg": [False, False]}  # 1,000 min
    alive = {"t": [0, 6], "s": [0.9, 0.1], "event_frame": None, "buds": [], "neg": [True, True]}
    r = LT.lead_times([flagged, never, capped, alive], 0.5)
    assert r["n_deaths"] == 3 and r["n_never_flagged"] == 1
    assert r["median_lead_min"] == 90.0                                         # of 90, 0 and the capped 960
    assert r["share_at_cap"] == pytest.approx(1 / 3)
    assert r["share_flagged_by"] == pytest.approx({"30": 2 / 3, "60": 2 / 3, "120": 1 / 3, "240": 1 / 3})
    assert r["false_alarm_rate"] == 0.25                                        # 1 of 4 negative windows
    assert r["alarms_per_100_negative_hours"] == 50.0                           # 1 alarm in 2 negative hours
    assert r["share_eventfree_lifespans_ever_alarm"] == 1.0
    assert r["median_buds_onset_to_death"] == 1.0                               # of 2 and 0
    alone = LT.lead_times([capped], 0.5)
    assert alone["median_lead_min"] == 960.0 and alone["share_at_cap"] == 1.0
    assert LT.median_lead_bootstrap([[capped, alive]], [0.5]) == [960.0, 960.0]  # the death of a resample, capped
    assert LT.median_lead_bootstrap([[never]], [0.5]) == [0.0, 0.0]
    assert LT.threshold(np.arange(101.0), 0.05) == pytest.approx(95.0)


def test_weighted_ap_is_auprc_on_the_resample():
    from deadahead.yeast.metrics import auprc
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.1).astype(int)
    s = np.round(rng.random(300), 2)
    prep = L._ap_prepare(y, s)
    assert L._weighted_ap(prep, np.ones(300)) == pytest.approx(auprc(y, s), abs=1e-15)
    w = rng.integers(0, 3, 300)
    assert L._weighted_ap(prep, w.astype(float)) == pytest.approx(auprc(np.repeat(y, w), np.repeat(s, w)), abs=1e-12)
    assert np.isnan(L._weighted_ap(L._ap_prepare(np.zeros(5), np.arange(5.0)), np.ones(5)))


def test_otsu_threshold_two_peaks():
    hist = np.zeros(256)
    hist[45:55] = hist[195:205] = 100
    assert D.otsu_threshold(hist) == 54                                         # the first k that splits the peaks
    hist = np.zeros(256)
    hist[10], hist[20], hist[200] = 50, 50, 50
    assert D.otsu_threshold(hist) == 20


def test_first_long_cycle_by_hand():
    buds = np.array([0, 10, 20, 30, 40, 50, 75, 85])                            # five cycles of 10, then 25 and 10
    f = L.detecdiv_features(90, 0, buds)
    assert f["age_min"] == 450 and f["divisions"] == 8
    assert f["since_last_bud_min"] == 25.0 and f["last_cycle_min"] == 50.0
    assert f["first_long_cycle"] == 1 and f["since_first_long_cycle_min"] == 75.0
    early = L.detecdiv_features(70, 0, buds)                                    # five closed cycles only
    assert early["first_long_cycle"] == 0 and np.isnan(early["since_first_long_cycle_min"])
    short = L.detecdiv_features(90, 0, np.array([0, 10, 20, 30, 40, 50, 69]))   # 19 min x 5 is under twice the median
    assert short["first_long_cycle"] == 0
    first = L.detecdiv_features(3, 3, np.zeros(0, int))
    assert first["since_last_bud_min"] == 0.0 and np.isnan(first["last_cycle_min"])


def test_single_scores_direction_and_missing_side():
    y = np.array([1, 1, 0, 0, 1, 1])
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.nan])
    s_tr, s_te, info = L.single_scores(x, y, np.array([2.0, np.nan]))
    assert info == {"direction": -1.0, "missing_side": "above"}                 # low values and missing mean death
    assert s_tr.tolist() == [-1.0, -2.0, -3.0, -4.0, 0.0, 0.0] and s_te.tolist() == [-2.0, 0.0]
    _, _, info = L.single_scores(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1]), np.array([1.0]))
    assert info == {"direction": 1.0, "missing_side": "below"}                  # a tie goes below
    assert L.single_scores(np.full(3, np.nan), np.array([0, 1, 0]), np.array([1.0]))[:2] == (None, None)


def test_sensitivity_series_by_hand():
    series = {"a": np.array([1, 1, L.EMPTY, L.EMPTY]), "b": np.array([1, 1, L.CLOG, L.CLOG]),
              "c": np.array([1, 1, L.DEAD, L.DEAD]), "d": np.array([1, 1, 1, 1])}
    changed = S.sensitivity_series(series, "empty_as_death")
    assert changed["a"].tolist() == [1, 1, L.DEAD, L.EMPTY] and changed["b"].tolist() == series["b"].tolist()
    assert L.detecdiv_lifespan(changed["a"]) == (0, 2, True)
    assert sorted(S.sensitivity_series(series, "clog_empty_left_out")) == ["c", "d"]
    with pytest.raises(ValueError):
        S.sensitivity_series(series, "other")


def test_population_by_hand():
    from deadahead.yeast import survival_scores as SS
    rows = pd.DataFrame({"unit": ["a", "b"], "t": [0, 6], "ev_off": [10.0, np.nan], "obs_off": [9.0, 20.0]})
    pop = SS.population(rows, {"a": 31, "b": 27})
    assert pop["T"].tolist() == [50.0, 100.0] and pop["delta"].tolist() == [1, 0]
    assert pop["C"].tolist() == [150.0, 100.0]                                  # a death: the end of its movie


def test_bud_death_bins_by_hand():
    #        a bud alone; death in an earlier bin; both in bin 1; censored in bin 2
    rows = pd.DataFrame({"nb_off": [3.0, 10.0, 4.0, np.nan], "ev_off": [np.nan, 4.0, 5.0, np.nan],
                         "obs_off": [50.0, 3.0, 4.0, 13.0]})
    out_bin, cause, cens = D.bud_death_bins(rows)
    assert out_bin.tolist() == [1.0, 1.0, 1.0, np.inf] and cause.tolist() == [1, 2, 2, 0]
    assert cens.tolist() == [np.inf, np.inf, np.inf, 2.0]
    ev_bin, cens_bin = D.survival_bins(pd.DataFrame({"ev_off": [6.0, 7.0, np.nan], "obs_off": [5.0, 6.0, 11.0]}))
    assert ev_bin.tolist() == [1.0, 2.0, np.inf] and cens_bin.tolist() == [np.inf, np.inf, 1.0]


def test_phase_groups_by_hand():
    rk = pd.DataFrame({"first_long_cycle": [1, 0, 0], "divisions": [9, 7, 6],
                       "since_last_bud_min": [59.0, 240.0, 241.0]})
    g = S.phase_groups(rk)
    assert g["phase"].tolist() == ["after_first_long_cycle", "before_first_long_cycle", "not_yet_decidable"]
    assert g["since_last_bud"].tolist() == ["under_1h", "1h_to_4h", "over_4h"]


def test_channel_inputs_read_the_training_positions_only(tmp_path):
    series = {f"{p}_1": np.ones(4, int) for p in C.POSITIONS}
    d = tmp_path / "vid_store"
    d.mkdir()
    rng = np.random.default_rng(0)
    for trap in series:
        v = np.full((4, *C.CROP_SHAPE), 100, dtype=np.uint8)                    # channel 2 alone has detail
        v[:, :, :, 2] = rng.integers(0, 256, size=(4, 60, 59), dtype=np.uint8)
        np.save(d / f"{trap}.npy", v)
    store = D.VidStore(d)
    by_pos = D.traps_by_position(series)
    held = C.POSITIONS[0]
    ch = D.choose_channel(store, by_pos, held)
    assert ch["channel"] == 2 and ch["n_frames"] == 24 and held not in ch["positions"]
    assert D.channel_moments(store, [f"{C.POSITIONS[1]}_1"], 0) == pytest.approx((100 / 255, 0.0))


def test_source_input_moments_leave_out_the_excluded_movie(tmp_path):
    from deadahead.video.model import PHASE_HI, PHASE_LO
    lo = np.full((3, 8, 8), PHASE_LO, dtype=np.float32)                         # maps to 0
    hi = np.full((3, 8, 8), PHASE_HI, dtype=np.float32)                         # maps to 1
    np.save(tmp_path / "a.npy", lo)
    np.save(tmp_path / "b.npy", hi)
    np.save(tmp_path / f"{D.SOURCE_EXCLUDED_MOVIES[0]}.npy", hi)
    m = D.source_input_moments(tmp_path)
    assert m["n_movies"] == 2 and m["n_frames"] == 6
    assert m["m_F"] == pytest.approx(0.5) and m["s_F"] == pytest.approx(0.5)


def test_load_vid_only_refuses_other_arrays():
    from scipy.io import savemat

    def mat(**arrays):
        buf = io.BytesIO()
        savemat(buf, arrays)
        return buf.getvalue()
    vid = np.zeros((*C.CROP_SHAPE, 5), dtype=np.uint8)
    assert D.load_vid_only(mat(vid=vid, deep=np.ones(5))).shape == (5, *C.CROP_SHAPE)    # the classes are never read
    with pytest.raises(D.StoreError):
        D.load_vid_only(mat(deep=np.ones(5)))
    with pytest.raises(D.StoreError):
        D.load_vid_only(mat(vid=vid.astype(float)))


def test_scorable_needs_all_positions_and_at_most_one_with_two_seeds():
    st = C.STARTS[0]
    missing = {(p, s) for p, s in zip(C.POSITIONS[:1], C.SEEDS[:1])}
    ok = lambda u: not any(S.run_name(st, p, s) == u for p, s in missing)       # noqa: E731
    assert S.survival_layer_scorable(st, ok)["scorable"]
    missing.add((C.POSITIONS[1], C.SEEDS[0]))
    assert not S.survival_layer_scorable(st, ok)["scorable"]
    assert not S.survival_layer_scorable(st, lambda u: C.POSITIONS[0] not in u)["scorable"]


def test_survival_scores_of_perfect_and_reversed_markers():
    from deadahead.yeast import survival_scores as SS
    ev = np.array([3.0, 5.0, 10.0, 20.0, 40.0, 45.0])                           # 15 .. 225 min after each landmark
    rows = pd.DataFrame({"unit": [f"u{i}" for i in range(6)], "t": 0, "ev_off": ev, "obs_off": ev - 1})
    pop = SS.population(rows, {f"u{i}": 1000 for i in range(6)})
    exact = {h: (pop["T"] <= h).astype(float) for h in C.HORIZONS}
    r = SS.scores(pop, exact)
    assert r["ibs_to_240"] == pytest.approx(0.0)
    # at 120 min: 8 pairs split by the horizon are concordant, the 6 pairs of deaths inside it tie at 0.5
    assert r["time_dependent_auc"]["120"] == pytest.approx(11 / 14)
    assert r["c_index_of_the_4h_risk"] == 0.5                                   # every 4-hour risk is 1: all tied
    ranked = {h: 1.0 - pop["T"] / 300 for h in C.HORIZONS}
    assert SS.scores(pop, ranked)["c_index_of_the_4h_risk"] == 1.0
    reverse = {h: pop["T"] / 300 for h in C.HORIZONS}
    assert SS.scores(pop, reverse)["c_index_of_the_4h_risk"] == 0.0


def test_next_bud_print_by_hand(tmp_path):
    p = C.POSITIONS[0]
    #        a bud within 30 min; later; none, death first; death first; undetermined; none, observed for 30 min
    nb = [3.0, 10.0, np.nan, 5.0, np.nan, np.nan]
    ev = [np.nan, np.nan, 2.0, 2.0, np.nan, np.nan]
    ob = [50.0, 50.0, 1.0, 1.0, 3.0, 10.0]
    rows = pd.DataFrame({"unit": [f"u{i}" for i in range(6)], "position": p, "t": 0, "nb_off": nb, "ev_off": ev,
                         "obs_off": ob, "y_30": [0, 0, 1, 1, np.nan, 0], "y_60": 0.0, "y_120": 0.0, "y_240": 0.0})
    run = S.run_name(C.STARTS[0], p, C.SEEDS[0])
    # the undetermined row has the highest bud score: counting it as a negative would lower the AUPRC
    fr = rows[["unit", "t"]].assign(cif_bud_30=[0.9, 0.1, 0.1, 0.1, 1.0, 0.1],
                                    cif_death_30=[0.1, 0.1, 0.9, 0.9, 0.1, 0.1])
    for h in C.HORIZONS[1:]:
        fr = fr.assign(**{f"cif_bud_{h}": 0.0, f"cif_death_{h}": 0.0})
    fr.to_parquet(tmp_path / "nb.parquet")
    out = S.next_bud_print(rows, lambda u: u == run, lambda u, kind: tmp_path / "nb.parquet")
    assert out[C.STARTS[0]][p]["30"] == {"bud": 1.0, "death": 1.0}
    assert out[C.STARTS[0]][p]["60"]["death"] is None                           # no death in that column
    assert out[C.STARTS[1]][p]["30"] == {"bud": None, "death": None}


def test_config_matches_the_constants():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "yeast.yaml").read_text())
    d = cfg["data"]
    assert tuple(d["positions"]) == C.POSITIONS and d["landmarks"] == C.N_LANDMARKS
    assert d["lifespans"] == C.N_LIFESPANS == sum(C.TRAPS_BY_POSITION.values()) - 1
    assert d["deaths"] == sum(C.DEATHS_BY_POSITION.values())
    assert cfg["rules"]["lightgbm"]["seed"] == L.LGB_SEED and cfg["rules"]["survival_head"]["seed"] == C.SEED


@pytest.mark.data
def test_the_numbers_of_s7_on_the_released_bundle():
    if not (S.default_bundle() / "MANIFEST.json").is_file():
        pytest.skip("the yeast bundle is not installed")
    s = S.score_bundle(S.default_bundle(), with_sensitivity=False, with_lead_time=False)["summary"]
    assert round(s["macro_auprc_30min"]["videomae_finetuned"], 4) == 0.0488
    assert round(s["macro_auprc_30min"]["videomae_public"], 4) == 0.0459
    assert round(s["macro_auprc_30min_best_rule"], 4) == 0.0394
    assert s["ahead_of_best_rule"]["videomae_finetuned"] == 4
    assert s["finetuned_ahead_of_public_start"] == 5


def test_survival_layer_tiny(fast_rules):
    from deadahead.yeast import survival as SV
    series = synthetic_series(n_traps=3, n_frames=200)
    rows, _ = D.build_rows(series)
    held = C.POSITIONS[0]
    keys = rows[["unit", "position", "t"]].reset_index(drop=True)
    emb = np.random.default_rng(0).normal(size=(len(keys), 8)).astype(np.float32)
    surv_rows = rows[rows["position"].isin(C.training_positions(held))]
    feats = rows[["unit", "position", "t", *L.DETECDIV_FEATURES]]
    pk = keys[keys["position"] == held].reset_index(drop=True)
    out = SV.fit_unit(emb, keys, surv_rows, feats, pk, C.SEED)
    assert set(out) == {"S", "SL", "NB"}
    cif = out["S"][[f"cif_{h}" for h in C.HORIZONS]].to_numpy()
    assert len(cif) == len(pk) and np.all(np.diff(cif, axis=1) >= -1e-12) and np.all((cif >= 0) & (cif <= 1))
    assert SV.parse_run("thr__videomae_public__Pos3_3")["kind"] == "thr"
    assert SV.parse_run("videomae_finetuned__Pos3_3__s2601563")["seed"] == 2601563
    with pytest.raises(ValueError):
        SV.parse_run("x__unknown_start__Pos3_3__s2601563")
