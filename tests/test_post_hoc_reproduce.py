"""The post hoc parts of ``deadahead reproduce`` (gap-window, alarm-rate) and the figure tables, on a synthetic bundle."""

import copy
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deadahead.evaluation.alarms import NOMINAL
from deadahead.evaluation.horizons import outcome_sets
from deadahead.evaluation.rows import CAUSES, GROUPS, KEY, LABEL_CENSORED, SEEDS, inner_groups, untyped_tracks
from deadahead import cli
from deadahead.report.figures import plot_alarm_rate, plot_gap_window, plot_warnings
from deadahead.reproduce import alarm_rate as A
from deadahead.reproduce import gap_window as W
from deadahead.reproduce.bundle import sha256_file
from deadahead.reproduce.compare import compare_tree
from deadahead.reproduce.gap import DEADAHEAD, DEEPHIT, HANDCRAFTED_SURVIVAL, IMAGE_SURVIVAL, gap_sets
from deadahead.reproduce.image_warnings import IMAGE, SPREAD_SERIES, SPREAD_STATS, spread_csv, spread_table
from deadahead.reproduce.main import PARTS, available_parts, figure_tables, main, run
from deadahead.reproduce.warning_times import TIME

N_FRAMES = 240
SURVIVAL = (DEADAHEAD, IMAGE_SURVIVAL, HANDCRAFTED_SURVIVAL, DEEPHIT)
ROOT = Path(__file__).resolve().parents[1]


def _tables(seed=5):
    """Nine groups, one 240-frame movie each, ten cells: three lytic deaths, three apoptoses, one untyped death and
    three cells alive at the end; a forecast every 10 frames from frame 40."""
    rng = np.random.default_rng(seed)
    rows, events = [], []
    for g in GROUPS:
        movie = f"{g}_m"
        for k in range(10):
            track = f"{movie}:{k}"
            kind = ("LYTIC", "LYTIC", "LYTIC", "APOPTOSIS", "APOPTOSIS", "APOPTOSIS", "UNTYPED_DEATH", "CENSORED",
                    "CENSORED", "CENSORED")[k]
            death = None if kind == "CENSORED" else int(rng.integers(110, 230))
            events.append({"track_id": track, "event_type": kind, "t_event": np.nan if death is None else float(death)})
            for t in range(40, N_FRAMES - 10, 10):
                if death is not None and t >= death:
                    break
                near_end = death is None and (N_FRAMES - 1 - t) * 3 < 30
                rows.append({"movie_id": movie, "track_id": track, "t_obs_frame": t, "group": g,
                             "label": LABEL_CENSORED if near_end else "NO_EVENT", "horizon_min": 30,
                             "window_weight": 1.0, "death": death, "kind": kind})
    return pd.DataFrame(rows), pd.DataFrame(events)


def _score(r, c, h, gen, off=0.0):
    """A risk of cause c within h min that is higher for the cells about to die of it, with noise."""
    lead = 3.0 * (r["death"].to_numpy(float) - r["t_obs_frame"].to_numpy(float))
    y = (r["kind"].to_numpy() == c) & (lead <= h)
    return np.clip(0.05 + 0.45 * y + 0.4 * gen.uniform(size=len(r)) + off, 0, 1)


def _bundle():
    rows, events = _tables()
    gen = np.random.default_rng(11)
    ci, ci2, pred, pred_img, inner, inner_img, inner_h = [], [], [], [], [], [], []
    for g in GROUPS:
        pg = rows[rows["group"] == g].reset_index(drop=True)
        key = {k: pg[k].to_numpy() for k in KEY}
        for s in SEEDS:
            for i, m in enumerate(SURVIVAL):
                ids = {"method": m, "group": g, "seed": s, **key}
                ci.append(pd.DataFrame({**ids, **{f"F_{c}_{h}": _score(pg, c, h, gen, 0.01 * i) for h in (60, 120)
                                                  for c in CAUSES}}))
                ci2.append(pd.DataFrame({**ids, **{f"F_{c}_{h}": _score(pg, c, h, gen, 0.01 * i) for h in (30, 240)
                                                   for c in CAUSES}}))
            for c in CAUSES:
                pred.append(pd.DataFrame({"group": g, "seed": s, "cause": c, **key, "score": _score(pg, c, 30, gen)}))
                pred_img.append(pd.DataFrame({"group": g, "seed": s, "cause": c, **key,
                                              "score": _score(pg, c, 30, gen, 0.02)}))
            for hh in inner_groups(g):
                ph = rows[rows["group"] == hh].reset_index(drop=True)
                ids = {"outer_group": g, "seed": s, "inner_group": hh, **{k: ph[k].to_numpy() for k in KEY}}
                inner_h.append(pd.DataFrame({**ids, **{f"F_{c}_{h}": _score(ph, c, h, gen) for h in (60, 120, 240)
                                                       for c in CAUSES}}))
                for c in CAUSES:
                    inner.append(pd.DataFrame({**ids, "cause": c, "score": _score(ph, c, 30, gen)}))
                    inner_img.append(pd.DataFrame({**ids, "cause": c, "score": _score(ph, c, 30, gen, 0.02)}))
    cols = ["movie_id", "track_id", "t_obs_frame", "group", "label", "horizon_min", "window_weight"]
    return {"rows": rows[cols], "events": events, "untyped": untyped_tracks(events),
            "movies": pd.DataFrame({"movie_id": [f"{g}_m" for g in GROUPS], "num_frames": N_FRAMES}),
            "landmarks": rows[KEY + ["group"]],
            "predictions": {DEADAHEAD: pd.concat(pred, ignore_index=True),
                            IMAGE: pd.concat(pred_img, ignore_index=True)},
            "inner_predictions": pd.concat(inner, ignore_index=True),
            "inner_predictions_by_method": {IMAGE: pd.concat(inner_img, ignore_index=True)},
            "cumulative_incidence": pd.concat(ci, ignore_index=True),
            "cumulative_incidence_30_240": pd.concat(ci2, ignore_index=True),
            "inner_cumulative_incidence": pd.concat(inner_h, ignore_index=True)}


@pytest.fixture(scope="module")
def parts():
    """The parts in the order --part all runs them."""
    b = _bundle()
    res = run(b, ("warnings", "image-warnings", "gap", "horizons", "gap-window", "alarm-rate"))
    return b, res


def _perturbed(tree: dict, *path, by=1e-8):
    out = copy.deepcopy(tree)
    d = out
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] += by
    return out


# ------------------------------------------------------------------ the window analysis
def _own_words(T, d, C, cause, g, H):
    """The sets written out without outcome_sets: E a death of either route at T <= g; P a death by the route with
    g < T <= H; Q one by the other route with g < T <= H; N no death by H and followed through H; U the rest."""
    k = {"APOPTOSIS": 1, "LYTIC": 2}[cause]
    dead = d != 0
    E = dead & (T <= g)
    P = (d == k) & (T > g) & (T <= H)
    Q = dead & (d != k) & (T > g) & (T <= H)
    N = ~(dead & (T <= H)) & (C >= H)
    return {"E": E, "P": P, "Q": Q, "N": N, "U": ~(E | P | Q | N)}


@pytest.mark.parametrize(("g", "H"), [(0, 120), (30, 120), (60, 120), (90, 120), (120, 240)])
def test_window_sets_match_their_definition_and_partition_the_rows(g, H):
    T = np.array([10.0, 40.0, 40.0, 70.0, 100.0, 100.0, 150.0, 150.0, 200.0, 500.0, 90.0])
    d = np.array([2, 2, 1, 1, 2, 0, 1, 2, 0, 0, 0])
    C = np.array([500.0, 500.0, 500.0, 500.0, 500.0, 100.0, 500.0, 500.0, 200.0, 500.0, 90.0])
    for c in CAUSES:
        s = W.window_sets(T, d, C, c, g, H)
        own = _own_words(T, d, C, c, g, H)
        for k in W.SET_NAMES:
            assert np.array_equal(s[k], own[k]), k
        assert (np.stack([s[k] for k in W.SET_NAMES]).astype(int).sum(axis=0) == 1).all()


def test_the_window_at_g0_is_the_horizon_and_at_g60_the_gap_analysis():
    T = np.array([30.0, 90.0, 90.0, 200.0, 50.0, 100.0])
    d = np.array([2, 2, 1, 0, 0, 0])
    C = np.array([500.0, 500.0, 500.0, 200.0, 50.0, 100.0])
    a, b = W.window_sets(T, d, C, "LYTIC", 60, 120), gap_sets(T, d, C, "LYTIC")
    h = outcome_sets(T, d, C, "LYTIC", 120)
    z = W.window_sets(T, d, C, "LYTIC", 0, 120)
    assert not z["E"].any()
    for k in ("P", "Q", "N", "U"):
        assert np.array_equal(z[k], h[k])
    for k in W.SET_NAMES:
        assert np.array_equal(a[k], b[k])


def test_a_gap_outside_the_window_is_refused():
    with pytest.raises(ValueError):
        W.window_sets(np.zeros(1), np.zeros(1, int), np.zeros(1), "LYTIC", 120, 120)


def test_the_scores_of_the_window_analysis():
    preds = {60: {"LYTIC": np.array([0.1, 0.2])}, 120: {"LYTIC": np.array([0.3, 0.5])}}
    assert np.array_equal(W.primary_score(DEADAHEAD, preds, "LYTIC", 120), [0.3, 0.5])
    assert np.allclose(W.secondary_score(DEADAHEAD, preds, "LYTIC", 60, 120), [0.2, 0.3])
    assert W.secondary_score(DEADAHEAD, preds, "LYTIC", 90, 120) is None
    assert np.array_equal(W.secondary_score(TIME, preds, "LYTIC", 90, 120), [0.3, 0.5])


def test_every_window_is_scored_on_the_same_rows(parts):
    _, res = parts
    out = res["gap-window"]
    assert list(out["windows"]) == ["120", "240"]
    assert out["post_hoc"] is True
    for H, by_g in W.WINDOWS.items():
        assert list(out["windows"][str(H)]) == [str(g) for g in by_g]
        for g in by_g:
            for c in CAUSES:
                blk = out["windows"][str(H)][str(g)]["routes"][c]
                assert blk["counts"]["total"]["rows"] == out["population"]["n_rows"]
                t = blk["counts"]["total"]
                assert t["E"] + t["P"] + t["Q"] + t["N"] + t["U"] == t["rows"]
                assert blk["counts"]["total"]["E"] == 0 or g > 0
                ch = sum(v["P"] / (v["P"] + v["Q"] + v["N"]) for v in blk["counts"]["per_group"].values()) / 9
                assert blk["chance"]["nine_folds"]["macro"] == pytest.approx(ch)


def test_the_printed_increase_is_named_where_the_forecast_is_not_stored(parts):
    _, res = parts
    w = res["gap-window"]["windows"]
    e = w["120"]["90"]["routes"]["LYTIC"]["methods"][DEADAHEAD]["secondary"]
    assert "not_stored" in e and "nine_folds" not in e
    assert "not_applicable" in w["120"]["0"]["routes"]["LYTIC"]["methods"][DEADAHEAD]["secondary"]
    assert "nine_folds" in w["120"]["30"]["routes"]["LYTIC"]["methods"][DEADAHEAD]["secondary"]
    assert "nine_folds" in w["240"]["120"]["routes"]["LYTIC"]["methods"][DEADAHEAD]["secondary"]
    t = w["120"]["90"]["routes"]["LYTIC"]["methods"][TIME]
    assert t["secondary"]["nine_folds"]["macro"] == t["primary"]["nine_folds"]["macro"]


def test_the_windows_agree_with_the_gap_and_horizons_parts(parts):
    _, res = parts
    c = res["gap-window"]["checks"]
    assert c["g0_equals_horizons_part"]["passes"] and c["g0_equals_horizons_part"]["n_checks"] > 0
    assert c["g60_equals_gap_part"]["passes"] and c["g60_equals_gap_part"]["n_checks"] == 6


def test_a_small_change_in_another_part_fails_the_window_checks(parts):
    b, res = parts
    g = GROUPS[0]
    gap = _perturbed(res["gap"], "routes", "LYTIC", "methods", DEADAHEAD, "primary", "nine_folds", "per_group", g)
    with pytest.raises(ValueError, match="g60_equals_gap_part"):
        W.part_gap_window(b, None, gap)
    hz = _perturbed(res["horizons"], "models", DEADAHEAD, "LYTIC", "120", "0.2_per_cell_hour", "auprc", "nine_folds",
                    "macro")
    with pytest.raises(ValueError, match="g0_equals_horizons_part"):
        W.part_gap_window(b, hz, None)


def test_a_row_without_a_score_makes_the_method_not_computed_for_the_window(parts):
    b, _ = parts
    pop = W.horizon_population(b)[0]
    ctx = W.context(b, pop)
    ctx["coll"][IMAGE_SURVIVAL][GROUPS[1]] = {"failed": "no file"}
    sets = W.window_set_table(ctx, 30, 120)
    scored, nc = W.method_status(ctx, sets, 120)
    assert IMAGE_SURVIVAL in nc and DEADAHEAD in scored
    blk = W.window_block(ctx, 30, 120)["routes"]["LYTIC"]
    assert blk["methods"][IMAGE_SURVIVAL]["primary"] is None
    assert "not_computed" in blk["contrasts"][f"{DEADAHEAD} - {IMAGE_SURVIVAL}"]


def test_the_window_table_and_figure(parts, tmp_path):
    _, res = parts
    text = W.figure_csv(res["gap-window"])
    (tmp_path / "t.csv").write_text(text)
    t = pd.read_csv(tmp_path / "t.csv")
    assert len(t) == 9 * 2 * 3
    assert set(t["method"]) == {DEADAHEAD, IMAGE_SURVIVAL, TIME}
    assert "H = 240 min, apoptosis" in W.format_gap_window(res["gap-window"])
    plot_gap_window(t, tmp_path / "t.png")
    assert (tmp_path / "t.png").stat().st_size > 1000


# ------------------------------------------------------------------ the alarm-rate analysis
def test_the_rates_are_exact_fractions():
    sw, gr = A.sweep_rates(), A.grid_rates()
    assert len(sw) == 16 and len(gr) == 160
    assert list(sw)[:2] == ["0.05_per_cell_hour", "0.1_per_cell_hour"] and list(sw)[-1] == "0.8_per_cell_hour"
    assert list(gr)[:2] == ["0.005_per_cell_hour", "0.01_per_cell_hour"] and list(gr)[-1] == "0.8_per_cell_hour"
    assert set(sw) <= set(gr)
    for rate, a in NOMINAL.items():
        assert sw[rate][1] == a
    assert gr["0.26_per_cell_hour"] == (Fraction(13, 50), Fraction(13, 100))
    with pytest.raises(ValueError):
        A.rate_name(Fraction(1, 3000))


def test_the_matched_rate_is_the_closest_with_its_tie_rule():
    c = {"a": (Fraction(1, 10), 0.30), "b": (Fraction(2, 10), 0.34), "c": (Fraction(3, 10), 0.26),
         "d": (Fraction(4, 10), float("nan"))}
    r = A.closest(c, 0.31)
    assert r["rate"] == "a" and r["n_non_finite_left_out"] == 1 and r["n_candidates"] == 3
    t = A.closest({"low": (Fraction(2, 10), 0.25), "high": (Fraction(1, 10), 0.75)}, 0.5)
    assert t["rate"] == "low" and [x["rate"] for x in t["tied_at_the_same_distance"]] == ["high"]
    same = A.closest({"x": (Fraction(2, 10), 0.5), "y": (Fraction(1, 10), 0.5)}, 0.4)
    assert same["rate"] == "y"
    assert "not_computed" in A.closest({"z": (Fraction(1, 10), float("nan"))}, 0.5)
    assert "not_computed" in A.closest({"z": (Fraction(1, 10), 0.5)}, float("nan"))


def test_the_rates_at_0_2_and_0_4_agree_with_the_warning_parts(parts):
    _, res = parts
    c = res["alarm-rate"]["checks"]["rates_0.2_and_0.4_equal_the_warning_parts"]
    assert c["passes"] and set(c["methods"]) == {DEADAHEAD, IMAGE, TIME}
    assert all(v["n_leaves_compared"] > 1000 and v["max_abs_diff"] == 0.0 for v in c["methods"].values())


def test_the_grid_equals_the_sweep_at_every_rate_both_hold(parts):
    _, res = parts
    c = res["alarm-rate"]["checks"]["grid_equals_the_sweep"]
    assert c["passes"] and c["n_shared"] == 16 and c["n_leaves_compared"] > 1000


def test_a_small_change_in_a_warning_part_fails_the_alarm_check(parts):
    b, res = parts
    w = _perturbed(res["warnings"], "models", DEADAHEAD, "LYTIC", "0.4_per_cell_hour", "sustained", "60", "nine_folds",
                   "macro")
    with pytest.raises(ValueError, match="rates_0.2_and_0.4"):
        A.part_alarm_rate(b, w, None)


def test_the_matched_comparison_follows_its_rule(parts):
    _, res = parts
    a = res["alarm-rate"]
    assert a["post_hoc"] is True
    target = a["sweep"][IMAGE]["LYTIC"]["0.2_per_cell_hour"]["achieved_rate_exposure_weighted"]["nine_folds"][
        "rate_per_cell_hour"]
    assert a["match_targets"]["c_star"]["LYTIC"] == target
    grid = a["grid"][DEADAHEAD]["LYTIC"]
    best = min(grid, key=lambda k: (abs(grid[k]["achieved_pooled"] - target), grid[k]["achieved_pooled"],
                                    grid[k]["r_per_cell_hour"]))
    m = a["matched"]["lytic"]
    assert m["match"]["rate"] == best and m["match"]["n_candidates"] == 160
    dead = m["deadahead_at_the_matched_rate"]["share_warned_last_60_min"]
    img = m["dinov2_handcrafted_at_0.2"]["share_warned_last_60_min"]
    diff = m["difference_deadahead_minus_image_last_60_min"]["nine_folds"]
    assert diff["macro_difference"] == pytest.approx(dead["macro"] - img["macro"])
    assert dead["eligible_deaths_per_group"] == img["eligible_deaths_per_group"]
    r = a["matched"]["reverse_lytic"]
    assert r["match"]["target"] == a["match_targets"]["deadahead_lytic_0.2"]
    assert a["matched"]["apoptosis"]["route"] == "APOPTOSIS"


def test_the_share_warned_rises_with_the_target_rate(parts):
    _, res = parts
    s = res["alarm-rate"]["sweep"][DEADAHEAD]["LYTIC"]
    v = [x["sustained"]["60"]["nine_folds"]["macro"] for x in s.values()]
    assert all(b >= a - 1e-12 for a, b in zip(v, v[1:])) and v[-1] > v[0]


def test_a_method_without_inner_predictions_is_refused():
    with pytest.raises(ValueError):
        A.part_alarm_rate({"inner_predictions": 1, "movies": 1})


def test_the_alarm_rate_table_and_figure(parts, tmp_path):
    _, res = parts
    text = A.figure_csv(res["alarm-rate"])
    (tmp_path / "t.csv").write_text(text)
    t = pd.read_csv(tmp_path / "t.csv")
    assert (t["set"] == "sweep").sum() == 16 * 3 * 2 and (t["set"] == "grid").sum() == 160 * 2 * 2
    assert (t["set"] == "matched").sum() == 2 and set(t.loc[t["set"] == "matched", "route"]) == {"LYTIC"}
    assert "Matched lytic comparison" in A.format_alarm_rate(res["alarm-rate"])
    plot_alarm_rate(t, tmp_path / "t.png")
    assert (tmp_path / "t.png").stat().st_size > 1000


# ------------------------------------------------------------------ Fig. 2 with its spread
def _warning_parts():
    """Fixed per-group shares per series, so the spread can be checked by hand."""
    def models(base):
        out = {}
        for c in CAUSES:
            sus = {X: {"nine_folds": {"per_group": {g: base + i / 100 + (c == "LYTIC") / 10
                                                    for i, g in enumerate(reversed(GROUPS))}}}
                   for X in ("30", "60", "90", "120", "180", "240")}
            out[c] = {"0.2_per_cell_hour": {"sustained": sus}}
        return out
    return {"models": {DEADAHEAD: models(0.5), TIME: models(0.1)}}, {"models": {IMAGE: models(0.3)}}


def test_the_spread_table_holds_the_mean_range_and_quartiles():
    warn, image = _warning_parts()
    row = spread_table(warn, image)[0]
    vals = 100 * (0.6 + np.arange(9) / 100)
    assert row["survival_lytic_r02"] == f"{vals.mean():.1f}"
    assert row["survival_lytic_r02_min"] == f"{vals.min():.1f}" and row["survival_lytic_r02_max"] == f"{vals.max():.1f}"
    assert row["survival_lytic_r02_q1"] == f"{np.percentile(vals, 25):.1f}"
    assert row["survival_lytic_r02_q3"] == f"{np.percentile(vals, 75):.1f}"
    assert row["clock_apop_r02_q3"] == f"{np.percentile(100 * (0.1 + np.arange(9) / 100), 75):.1f}"


def test_the_spread_csv_has_the_columns_of_the_stored_table():
    warn, image = _warning_parts()
    text = spread_csv(warn, image)
    head = "X," + ",".join(n + s for n, _, _ in SPREAD_SERIES for s in SPREAD_STATS)
    assert text.startswith(head + "\r\n") and text.count("\r\n") == 7
    stored = (ROOT / "docs" / "figures" / "data" / "fig2_warnings_spread.csv").read_bytes().decode()
    assert stored.splitlines()[0] == head and len(stored.splitlines()) == 7


def test_the_spread_needs_all_nine_groups():
    warn, image = _warning_parts()
    del warn["models"][DEADAHEAD]["LYTIC"]["0.2_per_cell_hour"]["sustained"]["60"]["nine_folds"]["per_group"][GROUPS[0]]
    with pytest.raises(ValueError):
        spread_table(warn, image)


def test_fig2_is_drawn_from_the_spread_table(tmp_path):
    warn, image = _warning_parts()
    (tmp_path / "t.csv").write_bytes(spread_csv(warn, image).encode())
    plot_warnings(pd.read_csv(tmp_path / "t.csv"), tmp_path / "t.png")
    assert (tmp_path / "t.png").stat().st_size > 1000


# ------------------------------------------------------------------ the parts table
def test_the_post_hoc_parts_run_after_the_parts_they_are_checked_against():
    for later, earlier in (("gap-window", "gap"), ("gap-window", "horizons"), ("alarm-rate", "warnings"),
                           ("alarm-rate", "image-warnings")):
        assert PARTS.index(earlier) < PARTS.index(later)


def test_the_parts_available_and_the_figure_tables(parts):
    b, res = parts
    have = available_parts(b)
    assert "gap-window" in have and "alarm-rate" in have
    assert set(figure_tables(res, "all")) == {"fig2_warnings.csv", "fig2_warnings_spread.csv",
                                              "figS8a_gap_window.csv", "figS8b_alarm_rate.csv"}
    assert set(figure_tables({"warnings": res["warnings"]}, "warnings")) == set()
    assert set(figure_tables({"gap-window": res["gap-window"]}, "gap-window")) == {"figS8a_gap_window.csv"}


def test_an_image_warnings_run_keeps_the_values_of_the_warnings_part(parts):
    b, res = parts
    alone = run(b, ("image-warnings",))
    assert set(alone) == {"warnings", "image-warnings"}
    assert compare_tree(alone["warnings"]["models"], res["warnings"]["models"], 0.0)["passes"]
    assert compare_tree(alone["image-warnings"]["models"], res["image-warnings"]["models"], 0.0)["passes"]


def _write_bundle(b: dict, folder: Path) -> None:
    """The synthetic bundle as a folder with its manifest."""
    files = {}

    def put(rel, frame, **info):
        p = folder / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(p, index=False)
        files[rel] = {"sha256": sha256_file(p), **info}

    for name in ("rows", "events", "movies", "landmarks", "inner_predictions", "cumulative_incidence",
                 "cumulative_incidence_30_240", "inner_cumulative_incidence"):
        put(f"{name}.parquet", b[name])
    put("predictions/deadahead.parquet", b["predictions"][DEADAHEAD], method=DEADAHEAD)
    put("predictions/image.parquet", b["predictions"][IMAGE], method=IMAGE)
    put("inner_predictions/image.parquet", b["inner_predictions_by_method"][IMAGE], method=IMAGE)
    (folder / "MANIFEST.json").write_text(json.dumps({"files": files}))


def test_the_command_line_writes_the_tables_and_draws_the_figures(parts, tmp_path):
    b, _ = parts
    _write_bundle(b, tmp_path / "bundle")
    out = tmp_path / "run"
    for part in ("gap-window", "image-warnings"):
        assert main(["--bundle", str(tmp_path / "bundle"), "--part", part, "--out", str(out / f"{part}.json")]) == 0
    for command, table in (("plot-gap-window", "figS8a_gap_window.csv"), ("plot-warnings", "fig2_warnings_spread.csv")):
        assert cli.main([command, str(out / table), str(out / f"{table}.png")]) == 0
        assert (out / f"{table}.png").stat().st_size > 1000
    assert (out / "fig2_warnings.csv").is_file()
    assert (out / "fig2_warnings_spread.csv").read_bytes().count(b"\r\n") == 7


def test_the_tree_comparison():
    assert compare_tree({"a": [1.0, "x"], "b": None}, {"a": [1.0, "x"], "b": None}, 0.0)["passes"]
    assert compare_tree({"a": float("nan")}, {"a": float("nan")}, 0.0)["passes"]
    assert not compare_tree({"a": float("nan")}, {"a": 1.0}, 0.0)["passes"]
    assert compare_tree({"a": 1.0}, {"a": 1.0 + 1e-13}, 1e-12)["passes"]
    assert not compare_tree({"a": 1.0}, {"a": 1.0 + 1e-8}, 1e-12)["passes"]
    assert not compare_tree({"a": 1.0}, {"b": 1.0}, 1e-12)["passes"]
    assert not compare_tree({"a": [1, 2]}, {"a": [1]}, 1e-12)["passes"]
    assert compare_tree({"a": 1.0, "skip": 2}, {"a": 1.0}, 0.0, ignore_keys=("skip",))["passes"]


def test_a_missing_block_or_a_changed_group_fails_the_block_comparison():
    block = {"nine_folds": {"macro": 0.5, "per_group": {"x": 0.5}}}
    assert W.compare_block(block, block) == []
    assert W.compare_block(block, None) == ["missing in the other part"]
    moved = {"nine_folds": {"macro": 0.5, "per_group": {"y": 0.5}}}
    assert W.compare_block(block, moved)
    assert W.compare_block(block, {"nine_folds": {"macro": float("nan"), "per_group": {"x": 0.5}}})


def test_the_post_hoc_config_matches_the_code():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "evaluation.yaml").read_text())
    gw, ar = cfg["gap_window"], cfg["alarm_rate"]
    assert {int(k): tuple(v) for k, v in gw["gaps_min"].items()} == W.WINDOWS
    assert tuple(gw["methods"]) == W.GAP_METHODS
    assert (ar["sweep_step"], ar["sweep_max"]) == (1 / A.SWEEP_DEN, A.SWEEP_STEPS / A.SWEEP_DEN)
    assert (ar["grid_step"], ar["grid_max"]) == (1 / A.GRID_DEN, A.GRID_STEPS / A.GRID_DEN)
    assert tuple(ar["methods"]) == A.ALARM_METHODS
