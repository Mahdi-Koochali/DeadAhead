import json

import numpy as np
import pytest

from deadahead.evaluation.survival_metrics import (APOPTOSIS, LYTIC, CensoringSurvivor, NoWeights, _jsonable,
                                                   _scorable, aalen_johansen, brier_complete_case, brier_ipcw,
                                                   calibration_intercept_slope, concordance, determined,
                                                   expected_over_observed, ipa, ipcw_intercept_slope, ipcw_weights,
                                                   metrics_block, null_brier, outcome, positivity)


def _world(n=400, seed=0, censor=None):
    """Competing-risks data: T exponential, cause 1 or 2, a marker that ranks the deaths; C the known end of
    observation (uniform when ``censor`` is given, else beyond every horizon)."""
    rng = np.random.default_rng(seed)
    risk = rng.uniform(0, 1, n)
    t_event = rng.exponential(300.0 / (0.2 + risk))
    cause = np.where(rng.uniform(size=n) < 0.5, APOPTOSIS, LYTIC)
    C = rng.uniform(0, censor, n) if censor else np.full(n, 1e6)
    T = np.minimum(t_event, C)
    delta = np.where(t_event <= C, cause, 0)
    return T, delta, C, risk


def _bruteforce_c(T, delta, M, cause, tau):
    num = den = 0.0
    for i in range(T.size):
        if delta[i] != cause or T[i] > tau:
            continue
        for j in range(T.size):
            if j == i:
                continue
            if (delta[j] not in (0, cause)) or T[j] > T[i] or (delta[j] == 0 and T[j] == T[i]):
                den += 1
                num += 1.0 if M[i] > M[j] else (0.5 if M[i] == M[j] else 0.0)
    return num / den


def test_outcome_and_determined():
    T = np.array([10.0, 50.0, 50.0, 300.0])
    d = np.array([1, 2, 0, 0])
    C = np.array([500.0, 500.0, 50.0, 300.0])
    assert outcome(T, d, APOPTOSIS, 30).tolist() == [1, 0, 0, 0]
    assert determined(T, d, C, 60).tolist() == [True, True, False, True]


def test_censoring_survivor_and_positivity():
    G = CensoringSurvivor([10.0, 20.0, 30.0, 300.0])
    assert np.allclose(G.left([5, 10, 11, 30, 31]), [1.0, 1.0, 0.75, 0.5, 0.25])
    p = positivity(G, (30, 60, 240))
    assert p["not_scorable_horizons"] == [] and p["n_C_below_240"] == 3
    p = positivity(CensoringSurvivor([10.0, 20.0, 30.0, 40.0, 300.0]), (30, 60))
    assert p["not_scorable_horizons"] == [60]
    with pytest.raises(ValueError):
        CensoringSurvivor([])


def test_aalen_johansen_without_censoring_is_the_empirical_incidence():
    T, d, _, _ = _world()
    for t in (30, 120, 240):
        aj = aalen_johansen(T, d, t)
        for c in (APOPTOSIS, LYTIC):
            assert aj[c] == pytest.approx(np.mean((d == c) & (T <= t)), abs=1e-12)


def test_weights_are_one_without_censoring_and_the_brier_scores_agree():
    T, d, C, risk = _world()
    G = CensoringSurvivor(C)
    w = ipcw_weights(T, d, C, 120, G)
    assert np.all(w == 1.0)
    p = np.clip(risk, 0, 1)
    assert brier_ipcw(T, d, C, p, LYTIC, 120, G) == pytest.approx(brier_complete_case(T, d, C, p, LYTIC, 120))


def test_null_prediction_has_zero_ipa():
    T, d, C, _ = _world(censor=900)
    G = CensoringSurvivor(C)
    nb = null_brier(T, d, C, LYTIC, 120, G)
    pred = np.full(T.size, nb["null_prediction"])
    assert ipa(brier_ipcw(T, d, C, pred, LYTIC, 120, G), nb["bs_null"]) == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(ipa(0.1, 0.0))


def test_concordance_without_censoring_matches_the_pairwise_count():
    T, d, C, risk = _world(n=150, seed=3)
    G = CensoringSurvivor(C)
    track = np.arange(T.size).astype(str)
    for cause in (APOPTOSIS, LYTIC):
        got = concordance(T, d, C, risk, cause, 240, G, track=track)["c_index"]
        assert got == pytest.approx(_bruteforce_c(T, d, risk, cause, 240), abs=1e-12)
        assert concordance(T, d, C, risk, cause, 240, NoWeights(), track=track)["c_index"] == pytest.approx(got)


def test_concordance_leaves_out_pairs_of_one_cell():
    T = np.array([30.0, 300.0, 300.0, 300.0])
    d = np.array([2, 0, 0, 0])
    C = np.array([500.0, 500.0, 300.0, 300.0])
    M = np.array([0.1, 0.9, 0.0, 0.0])
    G = CensoringSurvivor(np.full(4, 1e6))
    track = np.array(["a", "a", "b", "c"])
    assert concordance(T, d, C, M, LYTIC, 240, G, track=track)["c_index"] == 1.0
    assert concordance(T, d, C, M, LYTIC, 240, G, pair_set="all_pairs")["c_index"] == pytest.approx(2 / 3)


def test_ipcw_concordance_is_close_to_the_full_data_value_under_independent_censoring():
    rng = np.random.default_rng(11)
    n = 3000
    risk = rng.uniform(0, 1, n)
    t_event = rng.exponential(300.0 / (0.2 + risk))
    cause = np.where(rng.uniform(size=n) < 0.5, APOPTOSIS, LYTIC)
    full = concordance(t_event, cause, np.full(n, 1e6), risk, LYTIC, 120, CensoringSurvivor(np.full(n, 1e6)),
                       pair_set="all_pairs")["c_index"]
    C = rng.uniform(0, 700, n)
    T = np.minimum(t_event, C)
    d = np.where(t_event <= C, cause, 0)
    got = concordance(T, d, C, risk, LYTIC, 120, CensoringSurvivor(C), pair_set="all_pairs")["c_index"]
    assert abs(got - full) < 0.01


def test_calibration_of_calibrated_predictions():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.6, 20000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    a, b = calibration_intercept_slope(y, p)
    assert abs(a) < 0.15 and abs(b - 1) < 0.1
    T = np.where(y == 1, 10.0, 500.0)
    d = np.where(y == 1, LYTIC, 0)
    C = np.full(p.size, 500.0)
    a2, b2 = ipcw_intercept_slope(T, d, C, p, LYTIC, 30, CensoringSurvivor(C))
    assert a2 == pytest.approx(a, abs=1e-9) and b2 == pytest.approx(b, abs=1e-9)
    assert expected_over_observed(p, T, d, LYTIC, 30) == pytest.approx(p.mean() / y.mean())


def test_metrics_block_and_scorable_rules():
    import pandas as pd
    T, d, C, risk = _world(n=300, seed=7, censor=2000)
    pop = pd.DataFrame({"T_min": T, "delta": d, "C_min": C, "track_id": np.arange(T.size).astype(str),
                        "movie_id": "m", "group": "g"})
    preds = {h: {"APOPTOSIS": np.clip(risk * h / 400, 0, 1), "LYTIC": np.clip(risk * h / 300, 0, 1)}
             for h in (30, 240)}
    blk = metrics_block(pop, preds, (30, 240), None, ibs_too=True)
    fr = json.loads(json.dumps({"primary": {"population": blk}}, default=_jsonable))
    for c in ("APOPTOSIS", "LYTIC"):
        for h in (30, 240):
            v = _scorable(fr, "population", c, h, "ipa")
            assert v == blk["causes"][c]["horizons"][h]["ipa"]
            assert _scorable(fr, "population", c, h, "c_no_same_track") == \
                blk["causes"][c]["horizons"][h]["c"]["no_same_track"]["c_index"]
    fr["primary"]["population"]["G"]["not_scorable_horizons"] = [240]
    assert _scorable(fr, "population", "LYTIC", 240, "ipa") is None
    assert _scorable(fr, "population", "LYTIC", 240, "e_over_o") is not None
    fr["primary"]["population"]["causes"]["LYTIC"]["horizons"]["30"]["n_event_landmarks"] = 0
    assert _scorable(fr, "population", "LYTIC", 30, "e_over_o") is None


def test_a_determined_row_without_a_prediction_raises():
    T, d, C, risk = _world(n=50)
    p = risk.copy()
    p[0] = np.nan
    with pytest.raises(ValueError):
        brier_ipcw(T, d, C, p, LYTIC, 120, CensoringSurvivor(C))
