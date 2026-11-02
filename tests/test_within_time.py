"""The post hoc within-hour scoring (--part within-time), on synthetic rows and scores."""

import copy
import itertools
import json
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from deadahead.evaluation.aggregate import both_folds
from deadahead.evaluation.alarms import CUT_FRAME
from deadahead.evaluation.rows import CAUSES, GROUPS, KEY, SEEDS, table1_rows, untyped_tracks
from deadahead.evaluation.table1 import table1_unit
from deadahead.reproduce import within_time as W
from deadahead.reproduce.main import PARTS, available_parts, jsonable, report, run
from deadahead.reproduce.sensitivity import cut30
from deadahead.reproduce.table1 import TIME, part_table1
from deadahead.reproduce.warning_times import DEADAHEAD

L, A, E = "LYTIC", "APOPTOSIS", "NO_EVENT"
BLOCK = W.COMMON_CUT_BLOCK


# ------------------------------------------------------------------ oracles in their own words
def own_ap(y, s) -> float:
    """Average precision by hand: one operating point per distinct score, rows tied at a score enter together."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    total = 0.0
    for v in sorted(set(s.tolist()), reverse=True):
        called = s >= v
        total += (y & (s == v)).sum() / y.sum() * (y & called).sum() / called.sum()
    return float(total)


def own_within(score, label, bands, cause) -> float:
    b, y, s = np.asarray(bands), np.asarray(label) == cause, np.asarray(score, float)
    num = den = 0.0
    for u in sorted(set(b.tolist())):
        m = b == u
        if y[m].any():
            num += y[m].sum() * own_ap(y[m], s[m])
            den += y[m].sum()
    return float(num / den) if den else float("nan")


def exact_ap(in_order) -> Fraction:
    """The average precision of one strict order, best first, in exact fractions."""
    hits, total = 0, Fraction(0)
    for i, y in enumerate(in_order, 1):
        if y:
            hits += 1
            total += Fraction(hits, i)
    return total / hits


def exact_E(P: int, n: int) -> Fraction:
    if n == 1:
        return Fraction(1)
    h = sum(Fraction(1, i) for i in range(1, n + 1))
    return (Fraction(P - 1, n - 1) * (n - h) + h) / n


# ------------------------------------------------------------------ bands
def test_the_edges():
    assert W.EDGES["1h"] == tuple(range(40, 421, 20)) and W.EDGES["2h"] == tuple(range(40, 401, 40))
    assert W.EDGES["30min"] == tuple(range(40, 421, 10)) and W.BANDS == ("1h", "2h", "30min")
    assert W.PRIMARY_BANDS == "1h"


def test_each_forecast_time_lands_in_its_band():
    times = np.arange(40, 421, 10)
    one_hour = pd.Series(times).groupby(W.band_index(times, W.EDGES["1h"])).apply(list)
    assert all(len(v) == 2 for v in one_hour.iloc[:-1]) and one_hour.iloc[-1] == [420] and len(one_hour) == 20
    two_hours = pd.Series(times).groupby(W.band_index(times, W.EDGES["2h"])).apply(list)
    assert len(two_hours) == 10 and two_hours.iloc[0] == [40, 50, 60, 70] and two_hours.iloc[-1] == [400, 410, 420]
    assert len(set(W.band_index(times, W.EDGES["30min"]).tolist())) == times.size
    assert W.band_index([39, 40, 59, 60, 419, 420, 421], W.EDGES["1h"]).tolist() == [0, 1, 1, 2, 19, 20, 20]


@pytest.mark.parametrize("name", W.BANDS)
def test_the_bands_partition_the_rows(name):
    t = np.random.default_rng(3).choice(np.arange(40, 421, 10), 500)
    b = W.band_index(t, W.EDGES[name])
    assert (np.sum([b == u for u in np.unique(b)], axis=0) == 1).all()
    assert set(b.tolist()) <= set(range(1, len(W.EDGES[name]) + 1))
    c = W.band_counts(np.full(t.size, E), b, L)
    assert sum(v["n"] for v in c.values()) == t.size and all(v["P"] == 0 for v in c.values())


# ------------------------------------------------------------------ the weighted mean
def test_the_positive_weighted_mean_by_hand():
    # band 1: one death of 2 rows, ranked second (0.5); band 2: three deaths of 4 rows, ranked first (1)
    lab = np.array([L, E, L, L, L, E])
    bands = np.array([1, 1, 2, 2, 2, 2])
    score = np.array([0.1, 0.9, 0.9, 0.8, 0.7, 0.1])
    assert W.positive_weighted([0.5, 1.0], [1, 3]) == pytest.approx(0.875)
    assert W.within_band_value(score, lab, bands, L) == pytest.approx(0.875)
    assert W.within_band_value(score, lab, bands, L) == pytest.approx(own_within(score, lab, bands, L))
    assert np.isnan(W.positive_weighted([], []))


def test_a_band_without_a_death_contributes_nothing():
    lab = np.array([L, E, E, L, E, E, E, E, E, E])
    bands = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    rng = np.random.default_rng(5)
    score = rng.uniform(size=lab.size)
    base = W.within_band_value(score[:6], lab[:6], bands[:6], L)
    for _ in range(20):
        s = score.copy()
        s[6:] = rng.uniform(-5, 5, 4)
        assert W.within_band_value(s, lab, bands, L) == pytest.approx(base, abs=1e-15)
    assert W.band_counts(lab, bands, L)[3] == {"P": 0, "n": 4}
    full, part = W.band_chance(lab, bands, L), W.band_chance(lab[:6], bands[:6], L)
    assert full["random_order"] == part["random_order"] and full["positive_rate"] == part["positive_rate"]
    assert (full["rows_that_count"], full["bands_counted"], full["positives"]) == (6, 2, 2)


def test_no_death_at_all_is_undefined():
    assert np.isnan(W.within_band_value([0.1, 0.2], [E, E], [1, 2], L))
    assert np.isnan(W.band_chance(np.array([E, E]), [1, 2], L)["random_order"])


# ------------------------------------------------------------------ E(P, n)
@pytest.mark.parametrize("n", range(1, 10))
def test_E_against_enumeration(n):
    for P in range(1, n + 1):
        placements = list(itertools.combinations(range(n), P))
        exact = sum(exact_ap([i in pos for i in range(n)]) for pos in placements) / len(placements)
        assert exact == exact_E(P, n), (P, n)
        assert W.expected_auprc(P, n) == pytest.approx(float(exact), abs=1e-13), (P, n)
        table1_way = [table1_unit(-np.arange(n, dtype=float), np.array([L if i in pos else E for i in range(n)]), L)["auprc"]
                      for pos in placements]
        assert np.mean(table1_way) == pytest.approx(float(exact), abs=1e-12), (P, n)


def test_E_at_its_edges():
    assert W.expected_auprc(1, 1) == 1.0 and W.expected_auprc(1, 2) == pytest.approx(0.75)
    assert W.expected_auprc(40, 40) == pytest.approx(1.0) and W.harmonic(4) == pytest.approx(25 / 12)
    assert W.expected_auprc(7, 500) == pytest.approx(float(exact_E(7, 500)), abs=1e-13)
    assert W.expected_auprc(5, 800) > 5 / 800
    for P, n in ((0, 5), (6, 5), (1, 0)):
        with pytest.raises(ValueError, match="E\\(P, n\\)"):
            W.expected_auprc(P, n)


def test_band_chance_by_hand():
    lab = np.array([L, E, L, L, L, E, E, E, E])
    bands = np.array([1, 1, 2, 2, 2, 2, 3, 3, 3])
    ch = W.band_chance(lab, bands, L)
    assert ch["random_order"] == pytest.approx((float(exact_E(1, 2)) + 3 * float(exact_E(3, 4))) / 4, abs=1e-14)
    assert ch["positive_rate"] == pytest.approx((0.5 + 3 * 0.75) / 4, abs=1e-14)
    assert W.band_table(lab, bands, L) == {"band": [1, 2, 3], "P": [1, 3, 0], "n": [2, 4, 3]}
    # a death by the other route is a negative in n, and a censored row is not counted
    assert W.band_counts(np.array([L, A, "CENSORED_IN_HORIZON", E]), [1, 1, 1, 1], L) == {1: {"P": 1, "n": 3}}


# ------------------------------------------------------------------ what a score can do
@pytest.mark.parametrize("cause", CAUSES)
def test_a_perfect_score_reaches_one(cause):
    rng = np.random.default_rng(7)
    lab = rng.choice([L, A, E], 600, p=[0.1, 0.1, 0.8])
    t = rng.choice(np.arange(40, 421, 10), 600)
    y = lab == cause
    for name in W.BANDS:
        b = W.band_index(t, W.EDGES[name])
        assert W.within_band_value(y.astype(float), lab, b, cause) == pytest.approx(1.0, abs=1e-12)
        assert W.within_band_value(1.0 - y, lab, b, cause) < 1.0


def test_a_frame_only_score_equals_the_positive_rate_chance_under_unequal_30_min_bands():
    rng = np.random.default_rng(11)
    times = np.arange(40, 421, 10)
    t = np.repeat(times, rng.integers(1, 60, times.size))
    lab = rng.choice([L, A, E], t.size, p=[0.12, 0.08, 0.8])
    b = W.band_index(t, W.EDGES["30min"])
    for c in CAUSES:
        ch = W.band_chance(lab, b, c)
        for score in (t.astype(float), -t.astype(float), 1 / (1 + np.exp(-(t - 200.0) / 90.0))):
            assert W.within_band_value(score, lab, b, c) == ch["positive_rate"]
        cnt = [x for x in W.band_counts(lab, b, c).values() if x["P"]]
        assert ch["positive_rate"] != pytest.approx(np.mean([x["P"] / x["n"] for x in cnt]))


def test_a_drift_between_bands_changes_neither_design():
    rng = np.random.default_rng(13)
    lab = rng.choice([L, A, E], 800, p=[0.08, 0.06, 0.86])
    b = W.band_index(rng.choice(np.arange(40, 421, 10), 800), W.EDGES["1h"])
    s = rng.normal(size=800) + 1.0 * (lab == L)
    drifted = s + 10.0 * b
    assert W.within_band_value(drifted, lab, b, L) == pytest.approx(W.within_band_value(s, lab, b, L), abs=1e-12)
    assert W.pooled_value(drifted, lab, b, L) == pytest.approx(W.pooled_value(s, lab, b, L), abs=1e-12)
    assert table1_unit(drifted, lab, L)["auprc"] != pytest.approx(table1_unit(s, lab, L)["auprc"])


# ------------------------------------------------------------------ the rank rule and the second design
def test_the_rank_rule_with_ties_and_a_band_of_one_row():
    #        band 1: ties at 0.1      band 2: one row   band 3: tied throughout   band 4: three distinct
    score = [0.1, 0.1, 0.3, 0.9, 7.0, 2.0, 2.0, 2.0, 2.0, 5.0, 3.0, 1.0]
    bands = [1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 4, 4]
    want = [(1.5 - 1) / 3, (1.5 - 1) / 3, 2 / 3, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.0]
    assert W.rank_rule(score, bands) == pytest.approx(want, abs=1e-15)
    assert W.rank_rule([3.0], [1]).tolist() == [0.5]
    assert W.rank_rule(np.ones(1000), np.zeros(1000, int)).tolist() == [0.5] * 1000


def test_the_rank_rule_refuses_what_it_cannot_rank():
    with pytest.raises(ValueError, match="differ in shape"):
        W.rank_rule([1.0, 2.0], [1])
    with pytest.raises(ValueError, match="non-finite"):
        W.rank_rule([1.0, float("nan")], [1, 1])


def test_the_pooled_value_is_one_auprc_over_the_group_on_the_ranks():
    lab = np.array([L, E, E, L, E, E])
    bands = np.array([1, 1, 1, 2, 2, 2])
    score = np.array([0.9, 0.1, 0.2, 5.0, 7.0, 6.0])
    assert W.pooled_value(score, lab, bands, L) == pytest.approx(own_ap(lab == L, W.rank_rule(score, bands)))


def test_the_ceiling_by_hand_and_the_label_with_its_ties():
    # two bands of 3 rows with one death each: 1 at the top of the first band, 1 at the top of the second, then the rest
    lab = np.array([L, E, E, E, L, E])
    bands = np.array([1, 1, 1, 2, 2, 2])
    assert W.ceiling_value(lab, bands, L) == pytest.approx(1.0)
    # a band with no death still puts a row at the top, so an untied score falls short of 1
    lab = np.array([L, E, E, E, E, E])
    assert W.ceiling_value(lab, bands, L) == pytest.approx(own_ap(lab == L, W.rank_rule(W.deaths_first_score(lab, L), bands)))
    assert W.ceiling_value(lab, bands, L) == pytest.approx(0.5)
    assert W.tied_label_value(lab, bands, L) == pytest.approx(1.0)


def test_cut30_keeps_t_plus_10_up_to_the_cut():
    r = pd.DataFrame({"t_obs_frame": [410, 420, 428, 429, 430]})
    assert cut30(r)["t_obs_frame"].tolist() == [410, 420, 428]
    assert W.cut30 is cut30 and CUT_FRAME == 438


# ------------------------------------------------------------------ the part
def _tables(seed=5):
    """Nine groups, twelve cells each, a forecast every 10 frames from frame 40: four lytic deaths, four apoptoses, one
    untyped death, three cells alive at the end. A forecast is positive when the death comes within 30 min."""
    rng = np.random.default_rng(seed)
    rows, events = [], []
    for g in GROUPS:
        movie = f"{g}_m"
        for k in range(12):
            track = f"{movie}:{k}"
            kind = ("LYTIC", "LYTIC", "LYTIC", "LYTIC", "APOPTOSIS", "APOPTOSIS", "APOPTOSIS", "APOPTOSIS",
                    "UNTYPED_DEATH", "CENSORED", "CENSORED", "CENSORED")[k]
            death = None if kind == "CENSORED" else (435 if k in (3, 7) else int(rng.integers(100, 420)))
            events.append({"track_id": track, "event_type": kind, "t_event": np.nan if death is None else float(death)})
            for t in range(40, 431, 10):
                if death is not None and t >= death:
                    break
                near = death is not None and 3 * (death - t) <= 30
                label = kind if near else ("CENSORED_IN_HORIZON" if death is None and t == 430 else E)
                rows.append({"movie_id": movie, "track_id": track, "t_obs_frame": t, "group": g, "label": label,
                             "horizon_min": 30, "window_weight": 1.0})
    return pd.DataFrame(rows), pd.DataFrame(events)


def _bundle() -> dict:
    rows, events = _tables()
    gen = np.random.default_rng(11)
    preds = {}
    for i, m in enumerate(x for x in W.METHODS if x != TIME):
        out = []
        for g in GROUPS:
            pg = rows[rows["group"] == g]
            for s in SEEDS:
                for c in CAUSES:
                    y = (pg["label"] == c).to_numpy(float)
                    out.append(pd.DataFrame({"group": g, "seed": s, "cause": c, **{k: pg[k].to_numpy() for k in KEY},
                                             "score": (1.5 - 0.15 * i) * y + gen.normal(size=len(pg))}))
        preds[m] = pd.concat(out, ignore_index=True)
    return {"rows": rows, "events": events, "untyped": untyped_tracks(events), "predictions": preds}


@pytest.fixture(scope="module")
def world():
    b = _bundle()
    sens = {BLOCK: part_table1(b, rows_filter=cut30)}
    return b, sens, W.part_within_time(b, sens)


def test_the_part_is_post_hoc_and_every_check_ran_and_passed(world):
    res = world[2]
    assert res["post_hoc"] is True and res["not_computed"] == {}
    assert set(res["checks"]) == {"every_row_in_a_band", "positive_rate_with_one_band", "rank_rule_with_one_band",
                                  "elapsed_time_under_30_min_bands", "one_band_equals_the_sensitivity_part"}
    assert all(v["passes"] for v in res["checks"].values())
    assert res["checks"]["one_band_equals_the_sensitivity_part"]["n_checks"] == 2 * len(W.METHODS)


def test_alone_it_skips_only_the_checks_against_the_sensitivity_part(world):
    b, _, res = world
    alone = W.part_within_time(b)
    assert "one_band_equals_the_sensitivity_part" not in alone["checks"]
    assert alone["within_band"] == res["within_band"] and alone["pooled"] == res["pooled"]


def test_the_rows_are_the_cohort_at_the_common_cut_and_the_same_as_the_sensitivity_part(world):
    b, sens, res = world
    cohort = table1_rows(b["rows"], b["untyped"])["cohort"]
    assert res["rows"]["n_rows"] == len(cut30(cohort)) == sens[BLOCK]["cohort"]["n_rows"]
    assert res["rows"]["left_out"]["frames"] == [430] and res["rows"]["left_out"]["n"] == len(cohort) - res["rows"]["n_rows"]
    assert set(res["rows"]["left_out"]["labels"]) <= {L, A}


@pytest.mark.parametrize("name", W.BANDS)
def test_every_unit_equals_a_recomputation_in_its_own_words(world, name):
    b, _, res = world
    rows = cut30(table1_rows(b["rows"], b["untyped"])["cohort"])
    for m in (DEADAHEAD, W.IMAGE):
        d = b["predictions"][m]
        for g in ("DU145_bp", "PNT1A_st"):
            rg = rows[rows["group"] == g].reset_index(drop=True)
            bands = W.band_index(rg["t_obs_frame"], W.EDGES[name])
            for c in CAUSES:
                for s in SEEDS:
                    ds = d[(d["group"] == g) & (d["seed"] == s) & (d["cause"] == c)].set_index(KEY)["score"]
                    score = ds.reindex(pd.MultiIndex.from_frame(rg[KEY])).to_numpy()
                    got = res["within_band"][name][c]["methods"][m]["per_seed"][g][str(s)]
                    assert got == pytest.approx(own_within(score, rg["label"].to_numpy(), bands, c), abs=1e-12)


def test_aggregation_and_the_chance_beside_each_method(world):
    res = world[2]
    blk = res["within_band"]["1h"]["LYTIC"]
    e = blk["methods"][DEADAHEAD]
    means = {g: np.mean(list(v.values())) for g, v in e["per_seed"].items()}
    assert e["nine_folds"]["macro"] == pytest.approx(np.mean(list(means.values())), abs=1e-12)
    assert e["chance_beside"]["kind"] == "random_order"
    assert e["chance_beside"]["macro"] == pytest.approx(blk["chance"]["random_order"]["nine_folds"]["macro"], abs=1e-12)
    assert blk["methods"][TIME]["chance_beside"]["kind"] == "positive_rate"
    assert blk["methods"][TIME]["chance_beside"]["macro"] == pytest.approx(
        blk["chance"]["positive_rate"]["nine_folds"]["macro"], abs=1e-12)


def test_only_the_primary_contrast_decides_and_the_sentence_reads_it(world):
    res = world[2]
    for name, routes in res["within_band"].items():
        for c, blk in routes.items():
            assert set(blk["contrasts"]) == {f"{DEADAHEAD} - {m}" for m in W.OTHERS}
            for k, v in blk["contrasts"].items():
                want = "the primary contrast" if name == "1h" and k == W.PRIMARY_CONTRAST else "nothing"
                assert v["decides"] == want
    for c in CAUSES:
        assert all(v["decides"] == "nothing" for v in res["pooled"][c]["contrasts"].values())
    v, blk = res["paper_sentence_values"]["values"], res["within_band"]["1h"]["LYTIC"]
    assert v["x_L"] == blk["methods"][DEADAHEAD]["nine_folds"]["macro"]
    assert v["y_L"] == blk["chance"]["random_order"]["nine_folds"]["macro"]
    assert v["z_L"] == blk["methods"][W.IMAGE]["nine_folds"]["macro"]
    k = blk["contrasts"][W.PRIMARY_CONTRAST]["nine_folds"]
    assert v["d_L"] == k["macro_difference"] and v["k_L"] == k["x_ahead"]


def test_the_ceiling_and_the_label_as_a_tied_score(world):
    p = world[2]["pooled"]
    for c in CAUSES:
        assert 0 < p[c]["ceiling"]["nine_folds"]["macro"] <= 1
        assert p[c]["label_as_a_tied_score"]["nine_folds"]["macro"] == pytest.approx(1.0, abs=1e-12)
        assert set(p[c]["methods"]) == set(W.METHODS)


def _shift(sens, *path, by=1e-8):
    out = copy.deepcopy(sens)
    d = out[BLOCK]["cohort"]
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] += by
    return out


def test_a_small_change_in_the_sensitivity_part_stops_the_part(world):
    b, sens, _ = world
    with pytest.raises(ValueError, match="one_band_equals_the_sensitivity_part"):
        W.part_within_time(b, _shift(sens, "methods", DEADAHEAD, L, "nine_folds", "macro"))
    with pytest.raises(ValueError, match="one_band_equals_the_sensitivity_part"):
        W.part_within_time(b, _shift(sens, "methods", TIME, A, "nine_folds", "per_group", "LNCaP_do"))


def test_a_changed_count_in_the_sensitivity_part_stops_the_part(world):
    b, sens, _ = world
    units = sens[BLOCK]["cohort"]["methods"][DEADAHEAD][L]["units"]
    key = next(k for k in units if k.startswith(f"{GROUPS[0]}|"))
    changed = copy.deepcopy(sens)
    changed[BLOCK]["cohort"]["methods"] = {DEADAHEAD: changed[BLOCK]["cohort"]["methods"][DEADAHEAD]}
    changed[BLOCK]["cohort"]["methods"][DEADAHEAD][L]["units"][key]["N"] += 1
    with pytest.raises(ValueError, match="positive_rate_with_one_band"):
        W.part_within_time(b, changed)


def test_a_row_outside_the_bands_stops_the_part(world):
    b = world[0]
    rows = b["rows"].copy()
    rows.loc[rows.index[0], "t_obs_frame"] = 30
    with pytest.raises(ValueError, match="every_row_in_a_band"):
        W.part_within_time({**b, "rows": rows})


def test_the_rank_check_and_the_elapsed_time_check_can_fail():
    raw = {"m": {L: {"g": {1: 0.5}}}}
    assert W.check_rank_rule(raw, {"m": {L: {"g": {1: 0.5 + 1e-8}}}})["passes"] is False
    assert W.check_rank_rule(raw, raw)["passes"] is True
    lab = np.array([L, E, E, E])
    ctx = W.context(pd.DataFrame({"group": GROUPS[0], "t_obs_frame": [40, 40, 50, 50], "label": lab}))
    tied = {TIME: {(GROUPS[0], 0): {c: np.array([1.0, 1.0, 2.0, 2.0]) for c in CAUSES}}}
    assert W.check_elapsed_time(ctx, tied)["passes"] is True
    untied = {TIME: {(GROUPS[0], 0): {c: np.array([2.0, 1.0, 2.0, 2.0]) for c in CAUSES}}}
    assert W.check_elapsed_time(ctx, untied)["passes"] is False


def test_a_missing_method_is_named_and_a_missing_baseline_stops_the_part(world):
    b = world[0]
    fewer = {k: v for k, v in b["predictions"].items() if k != "dinov2|refit"}
    res = W.part_within_time({**b, "predictions": fewer})
    assert res["not_computed"] == {"dinov2|refit": "not in the bundle"}
    assert "dinov2|refit" not in res["within_band"]["1h"][L]["methods"]
    no_image = {k: v for k, v in b["predictions"].items() if k != W.IMAGE}
    with pytest.raises(ValueError, match="needs DeadAhead and DINOv2"):
        W.part_within_time({**b, "predictions": no_image})
    assert "within-time" not in available_parts({**b, "predictions": no_image})


def test_a_method_that_cannot_cover_the_rows_is_never_partly_scored(world):
    b = world[0]
    p = dict(b["predictions"])
    d = p["dinov2|refit"]
    p["dinov2|refit"] = d[d["t_obs_frame"] != 40]
    res = W.part_within_time({**b, "predictions": p})
    assert "evaluated rows have no prediction" in res["not_computed"]["dinov2|refit"]
    assert "dinov2|refit" not in res["summary"]["within_band"]["2h"][A]["methods"]


def test_the_part_runs_after_the_sensitivity_part_and_prints_its_scale(world):
    b, _, _ = world
    assert PARTS.index("sensitivity") < PARTS.index("within-time") and "within-time" in available_parts(b)
    res = run(b, ("within-time",))
    text = report(res)
    assert text.startswith("Post hoc within-hour scoring") and "not Table 1's" in text
    assert "Post hoc within-hour AUPRC, lytic: DeadAhead" in text and "a random order" in text and "deciding nothing" in text
    assert json.loads(json.dumps(res, default=jsonable))["within-time"]["post_hoc"] is True


def test_a_group_without_a_death_is_left_out_and_named(world):
    b = world[0]
    rows = b["rows"].copy()
    rows.loc[(rows["group"] == "LNCaP_do") & (rows["label"] == L), "label"] = E
    res = W.part_within_time({**b, "rows": rows})
    e = res["within_band"]["1h"][L]["methods"][DEADAHEAD]
    assert e["nine_folds"]["n_groups"] == 8 and "LNCaP_do" in e["nine_folds"]["groups_left_out"]
    assert e["chance_beside"]["n_groups"] == 8 and "LNCaP_do" not in e["chance_beside"]["groups"]
    assert res["within_band"]["1h"][A]["methods"][DEADAHEAD]["nine_folds"]["n_groups"] == 9


def test_the_one_band_value_is_table_1s(world):
    b, sens, _ = world
    rows = cut30(table1_rows(b["rows"], b["untyped"])["cohort"])
    ctx = W.context(rows)
    scores, _ = W.method_scores(b, rows)
    units = W.scored_units(ctx, {DEADAHEAD: scores[DEADAHEAD]}, W.within_band_value)
    want = sens[BLOCK]["cohort"]["methods"][DEADAHEAD][A]["nine_folds"]
    assert both_folds(units[DEADAHEAD][A])["nine_folds"]["macro"] == pytest.approx(want["macro"], abs=1e-15)
