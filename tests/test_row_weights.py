"""The row weights reach every fit: the LightGBM datasets, the logistic regression and the loss of each network."""

import numpy as np
import pandas as pd
import pytest

from deadahead.baselines import elapsed_time as ET
from deadahead.baselines import heads as H
from deadahead.survival import censoring as CE
from deadahead.survival import head
from tests.test_survival_head import CFG, CFG_W, COLS, synthetic_landmarks


@pytest.fixture
def datasets(monkeypatch):
    """Every LightGBM dataset built during a test: (label, weight)."""
    seen = []
    real = head.lgb.Dataset

    def recording(data, label=None, weight=None, **kw):
        seen.append((np.asarray(label), None if weight is None else np.asarray(weight, dtype=float)))
        return real(data, label=label, weight=weight, **kw)
    monkeypatch.setattr(head.lgb, "Dataset", recording)
    return seen


def test_survival_head_repeats_the_landmark_weight_in_every_bin_at_risk(datasets, monkeypatch):
    monkeypatch.setattr(head, "ROUNDS", 5)
    head.set_threads(1)
    tr = synthetic_landmarks(300, 0)
    tr["window_weight"] = np.random.default_rng(1).uniform(0.1, 3.0, len(tr))
    head.train_predict_survival(tr, synthetic_landmarks(20, 1), COLS, CFG, CFG_W, 7)
    dt = (tr["t_event_true"] - tr["t_obs_frame"]).to_numpy()
    ev_bin = np.where(tr["true_event"].isin(head.CAUSES) & (dt > 0), np.ceil(dt / 10), np.inf)
    c = tr["censor_offset"].to_numpy()
    cens_bin = np.where(np.isfinite(c), np.ceil(c / 10), np.inf)
    want = np.concatenate([tr["window_weight"].to_numpy()[(ev_bin >= k) & (cens_bin >= k)] for k in range(1, 9)])
    assert len(datasets) == 1 and np.array_equal(datasets[0][1], want)


def test_lightgbm_heads_read_the_window_weight(datasets, monkeypatch):
    monkeypatch.setattr(H, "ROUNDS", 5)
    H.set_threads(1)
    from tests.test_baselines_heads import synthetic_windows
    w, hist = synthetic_windows()
    rows, lm = H.forecast_rows(w)
    data, _ = H.build_table(rows, lm, hist, None)
    H.refit_fold(data, ["f0", "f1", "f2"], "LNCaP_do", [1], drop_untyped=True)
    tr = data[(data["horizon_min"] == 30) & (data["group"] != "LNCaP_do") & (data["label"] != H.UNTYPED)]
    assert len(datasets) == 2
    for _, wt in datasets:
        assert np.array_equal(wt, tr["window_weight"].to_numpy())


def test_censoring_classifiers_read_the_window_weight(datasets, monkeypatch):
    monkeypatch.setattr(CE, "ROUNDS", 5)
    CE.set_threads(1)
    rng = np.random.default_rng(0)
    n = 400
    tr = pd.DataFrame({"x": rng.normal(size=n), "t_obs_frame": 40, "t_event_true": np.nan, "censor_offset": np.inf,
                       "oracle_label": rng.integers(0, 3, n), "window_weight": rng.uniform(0.1, 3.0, n)})
    CE.train_predict_horizon(tr, tr.head(5), ["x"], "LYTIC", "c_censoring_aware", 10, 3)
    assert len(datasets) == 1 and np.array_equal(datasets[0][1], tr["window_weight"].to_numpy())


def test_elapsed_time_is_weighted():
    t = pd.DataFrame({"time_in_movie_min": [0.0, 0.0, 60.0, 60.0], "label": ["LYTIC", "NONE", "LYTIC", "NONE"],
                      "window_weight": [3.0, 1.0, 1.0, 3.0]})
    p = ET.fit_predict(t, t, 2, "LYTIC")
    assert p[0] == pytest.approx(0.75, abs=1e-4) and p[2] == pytest.approx(0.25, abs=1e-4)   # 3 / (3 + 1), 1 / (1 + 3)
