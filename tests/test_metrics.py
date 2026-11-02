import math

import numpy as np
import pandas as pd
import pytest

from deadahead.baselines.elapsed_time import fit_predict, prepare
from deadahead.evaluation.aggregate import (AHEAD_X, AHEAD_Y, NOT_IMPORTANT, NOT_SCORABLE, both_folds, macro,
                                            paired)
from deadahead.evaluation.metrics import auprc
from deadahead.evaluation.rows import GROUPS


def brute_force_ap(y, s):
    """Average precision from its definition: walk the distinct thresholds from the highest score down."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    n_pos = y.sum()
    ap, prev_recall = 0.0, 0.0
    for t in sorted(set(s.tolist()), reverse=True):
        called = s >= t
        tp = (y & called).sum()
        recall = tp / n_pos
        ap += (recall - prev_recall) * (tp / called.sum())
        prev_recall = recall
    return ap


@pytest.mark.parametrize("seed", range(20))
def test_auprc_matches_brute_force_with_ties(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 200))
    y = rng.random(n) < rng.uniform(0.05, 0.5)
    y[0] = True
    s = np.round(rng.random(n), int(rng.integers(1, 3)))     # few decimals: many ties
    assert auprc(y.astype(int), s) == pytest.approx(brute_force_ap(y, s), abs=1e-12)


def test_auprc_constant_score_is_prevalence_whatever_the_order():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for perm in (y, y[::-1], np.roll(y, 5)):
        assert auprc(perm, np.zeros(10)) == pytest.approx(0.1)


def test_auprc_perfect_and_undefined():
    assert auprc(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])) == 1.0
    assert math.isnan(auprc(np.zeros(4, int), np.arange(4.0)))


def test_macro_is_mean_over_groups_of_seed_means():
    rng = np.random.default_rng(0)
    pgs = {g: {s: float(rng.random()) for s in (1, 2, 3)} for g in GROUPS}
    m = macro(pgs)
    seed_means = [np.mean(list(pgs[g].values())) for g in GROUPS]
    assert m["macro"] == pytest.approx(np.mean(seed_means), abs=1e-15)
    assert m["n_groups"] == 9 and m["n_left_out"] == 0
    for g in GROUPS:
        assert m["per_group"][g] == pytest.approx(np.mean(list(pgs[g].values())), abs=1e-15)


def test_macro_leaves_out_a_group_with_an_undefined_seed():
    pgs = {g: {1: 0.5, 2: 0.5} for g in GROUPS}
    pgs["LNCaP_bp"][2] = float("nan")
    m = macro(pgs)
    assert m["n_groups"] == 8 and "LNCaP_bp" in m["groups_left_out"]
    b = both_folds(pgs)
    assert b["eight_folds_without_PNT1A_bp"]["n_groups"] == 7
    assert "PNT1A_bp" not in b["eight_folds_without_PNT1A_bp"]["groups_contributing"]


def _pair(diffs):
    y = {g: 0.5 for g in GROUPS[:len(diffs)]}
    x = {g: 0.5 + d for g, d in zip(GROUPS, diffs)}
    return x, y


def test_decision_rule_outcomes():
    x, y = _pair([0.05] * 7 + [-0.01] * 2)                  # macro +0.0367, 7 of 9 ahead
    assert paired(x, y)["outcome"] == AHEAD_X
    x, y = _pair([-0.05] * 8 + [0.01])
    assert paired(x, y)["outcome"] == AHEAD_Y
    x, y = _pair([0.05] * 6 + [-0.001] * 3)                 # large macro, only 6 of 9 ahead
    assert paired(x, y)["outcome"] == NOT_IMPORTANT
    x, y = _pair([0.01] * 9)                                # 9 of 9 ahead, macro below 0.02
    assert paired(x, y)["outcome"] == NOT_IMPORTANT
    x, y = _pair([0.05] * 6)                                # only 6 groups on both sides
    r = paired(x, y)
    assert r["outcome"] == NOT_SCORABLE and r["n"] == 6
    x, y = _pair([0.05] * 7 + [0.0] * 2)
    r = paired(x, y)
    assert (r["x_ahead"], r["y_ahead"], r["ties"]) == (7, 0, 2)


def test_elapsed_time_logistic_ranks_late_rows_higher_when_deaths_are_late():
    t = np.arange(400)
    w = pd.DataFrame({"t_obs_frame": t, "label": np.where(t > 300, "LYTIC", "NO_EVENT"),
                      "window_weight": 1.0, "horizon_min": 30, "group": "A"})
    e = prepare(w)
    p = fit_predict(e, e, 2, "LYTIC")
    assert np.all(np.diff(p) >= 0) and p[-1] > 0.5 > p[0]


def test_evaluation_config_matches_the_code_constants():
    from pathlib import Path

    import yaml

    from deadahead.evaluation import aggregate as A
    from deadahead.evaluation import alarms as W
    from deadahead.evaluation import rows as R
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "evaluation.yaml").read_text())
    assert tuple(cfg["groups"]) == R.GROUPS and tuple(cfg["seeds"]) == R.SEEDS and tuple(cfg["causes"]) == R.CAUSES
    assert cfg["table1"]["horizon_min"] == R.TABLE1_HORIZON
    assert cfg["table1"]["eight_fold_print_without"] == A.LEFT_OUT_FOLD
    assert (cfg["decision_rule"]["margin"], cfg["decision_rule"]["count"], cfg["decision_rule"]["min_groups"]) == (
        A.MARGIN, A.COUNT, A.MIN_GROUPS)
    w = cfg["warnings"]
    assert w["score_horizon_min"] == W.WARNING_HORIZON and w["follow_min"] == W.FOLLOW_MIN
    assert w["row_hours"] == W.ROW_HOURS and w["cut_frame"] == W.CUT_FRAME and w["minutes_per_frame"] == W.FRAME_MIN
    assert {k: float(v) for k, v in W.NOMINAL.items()} == w["nominal_alpha"]
    assert tuple(w["sustained_x_min"]) == W.SUSTAINED_X and w["elapsed_time_tie_seed"] == W.TIE_SEED
    from deadahead.evaluation import survival_metrics as M
    assert tuple(cfg["horizons"]["horizons_min"]) == M.HORIZONS_MIN
    assert cfg["horizons"]["positivity_min_g"] == M.POSITIVITY_MIN_G


def test_alarm_threshold_counts_ties_against_the_alarm():
    from fractions import Fraction

    from deadahead.evaluation.alarms import threshold_and_apply
    ref = np.arange(1.0, 21.0)                              # 20 reference rows, alpha 1/10: k = 2
    flags, thr = threshold_and_apply(ref, np.array([20.0, 19.0, 18.5, 18.0, 0.0]), Fraction(1, 10))
    assert thr["k"] == 2 and thr["n_negatives"] == 20
    assert flags.tolist() == [True, True, True, False, False]   # 18.5: two reference rows score >= it
    tied = np.r_[np.arange(1.0, 18.0), 19.0, 19.0, 19.0]   # a tie of three straddles the cut
    flags, thr = threshold_and_apply(tied, np.array([19.0, 19.5]), Fraction(1, 10))
    assert flags.tolist() == [False, True] and thr["cut_tie"]["n_negatives_tied"] == 3


def test_sustained_coverage_and_median_warning():
    from fractions import Fraction

    from deadahead.evaluation.alarms import median_sustained_warning, sustained
    lead = np.array([15.0, 45, 75, 105, 135, 165, 195, 225] * 2)      # two deaths, one row per 30-min bin
    track = np.array(["a"] * 8 + ["b"] * 8)
    cause = np.ones(16, int)
    flags = np.array([1, 1, 1, 0, 1, 1, 1, 1] + [0, 1, 1, 1, 1, 1, 1, 1], bool)
    s = sustained(flags, track, lead, cause, 1, Fraction(1, 10))
    assert [s[x]["share"] for x in ("30", "60", "90", "120", "180", "240")] == [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
    assert s["60"]["chance_lower"] == pytest.approx(0.01) and s["60"]["n_eligible_events"] == 2
    med = median_sustained_warning(flags, track, lead, cause, 1)
    assert med["n_events_all_eight_bins"] == 2 and med["median_min"] == 45.0      # runs of 90 and 0 min


def test_elapsed_time_score_does_not_depend_on_row_order():
    from deadahead.evaluation.alarms import time_scores
    rows = pd.DataFrame({"movie_id": ["m"] * 6, "track_id": list("aabbcc"), "t_obs_frame": [0, 10] * 3})
    a = time_scores(rows)
    shuffled = rows.sample(frac=1.0, random_state=3)
    b = time_scores(shuffled)
    assert np.array_equal(a.to_numpy(), b.loc[rows.index].to_numpy())
    assert np.all(np.floor(a.to_numpy()) == rows["t_obs_frame"].to_numpy())
