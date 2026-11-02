import numpy as np
import pandas as pd
import pytest

from deadahead.evaluation import censoring as E
from deadahead.evaluation.rows import GROUPS
from deadahead.reproduce.censoring import format_censoring, part_censoring, unit_payloads
from deadahead.survival import censoring as C

lgb = pytest.importorskip("lightgbm")


# ------------------------------------------------------------------ scoring
def synthetic_units(rng, regimes=E.REGIMES, seeds=(1, 2, 3)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every regime scores the label with noise; c_censoring_aware has the least noise."""
    noise = {"a_naive_drop": 4.0, "b_naive_mislabel": 3.0, "c_censoring_aware": 0.1, "d_joint_survival": 2.0}
    preds, units = [], []
    for g in GROUPS:
        y = {h: rng.choice([0, 0, 0, 1, 2], size=40) for h in E.HORIZONS}
        for r in regimes:
            for s in seeds:
                for h in E.HORIZONS:
                    yy = y[h]
                    d = {"group": g, "seed": s, "regime": r, "horizon_min": h, "label": yy}
                    for c in E.CAUSES:
                        d[f"score_{c}"] = (yy == E.CAUSE_IX[c]) + noise[r] * rng.random(len(yy))
                    preds.append(pd.DataFrame(d))
                row = {"group": g, "seed": s, "regime": r, "achieved_censoring_rate": 0.4}
                for k in ("training_landmarks", "bins_at_risk_original", "bins_at_risk_corrected", "bins_removed",
                          "landmarks_truncated"):
                    row[k] = 3 if r == "d_joint_survival" else -1
                for k in ("censor_bin_at_risk", "labelled_survived", "labelled_with_a_later_death_in_that_bin",
                          "event_rows"):
                    row[f"injected_{k}"] = 2 if r == "d_joint_survival" else -1
                units.append(row)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(units)


def test_part_censoring_contrasts():
    preds, units = synthetic_units(np.random.default_rng(0))
    res = part_censoring({"censoring_predictions": preds, "censoring_units": units})
    assert res["units"] == len(GROUPS) * 4 * 3
    v = res["contrasts"]["c_censoring_aware_minus_a_naive_drop|LYTIC|h30"]
    assert v["n_units"] == len(GROUPS) and v["wins"] == len(GROUPS) and v["mean_diff"] > 0
    assert res["paper_paragraph"]["values"]["LYTIC"]["groups_ahead"] == len(GROUPS)
    assert res["results_by_regime"]["c_censoring_aware"]["APOPTOSIS"]["240"]["n_groups"] == len(GROUPS)
    assert res["correction"]["training_bins_removed_total"] == 3 * len(GROUPS) * 3
    assert res["correction"]["injected_censor_bin"]["labelled_with_a_later_death_in_that_bin_share_of_event_rows"] == 1
    assert "c_censoring_aware_minus_b_naive_mislabel" in format_censoring(res)


def test_unit_payloads_keep_order():
    preds, _ = synthetic_units(np.random.default_rng(1), regimes=("a_naive_drop",), seeds=(7,))
    p = unit_payloads(preds)[(GROUPS[0], 7, "a_naive_drop")]
    d = preds[(preds["group"] == GROUPS[0]) & (preds["horizon_min"] == 60)]
    assert np.array_equal(p["y_60"], d["label"].to_numpy())
    assert np.array_equal(p["p_a_naive_drop_LYTIC_60"], d["score_LYTIC"].to_numpy())


def test_paper_side_withholds_short_contrasts():
    assert E.paper_side({"mean_diff": 0.1, "wins": 3, "n_units": 8}, 9)["mean_diff"] is None
    assert E.paper_side({"mean_diff": 0.1, "wins": 3, "n_units": 9}, 9)["groups_ahead"] == 3


# ------------------------------------------------------------------ training
def tiny_data(rng, n_groups: int = 3) -> tuple[pd.DataFrame, list]:
    """Oracle rows joined to two features for three groups; one untyped-death cell."""
    out = []
    for gi, g in enumerate(GROUPS[:n_groups]):
        for tr in range(40):
            fate = ["APOPTOSIS", "LYTIC", "NONE", "NONE"][tr % 4]
            t_ev = 140.0 if fate != "NONE" else np.nan
            for t in range(40, 140, 10):
                for h in (30, 60, 120, 240):
                    end = t + h // 3
                    within = fate != "NONE" and t < t_ev <= end
                    clean = not within and (t_ev if fate != "NONE" else 200) >= end
                    if not (within or clean):
                        continue
                    out.append({"movie_id": f"{g}_1", "track_id": f"{g}:{tr}", "t_obs_frame": t, "group": g,
                                "treatment": "bp", "window_weight": 1.0, "true_event": fate,
                                "t_event_true": t_ev, "t_end_true": t_ev if fate != "NONE" else 199.0,
                                "horizon_min": h, "oracle_label": C.CAUSE_IX.get(fate, 0) if within else 0,
                                "f1": (fate == "LYTIC") + rng.normal(0, 0.3), "f2": (fate == "APOPTOSIS")
                                + rng.normal(0, 0.3)})
    return pd.DataFrame(out), ["f1", "f2"]


CFG = {"survival_grid": {"bin_frames": 10, "n_bins": 8}}
CFG_W = {"frame_interval_min": 3.0, "horizons_min": [30, 60, 120, 240]}


def test_inject_censor_time_hits_the_rate():
    data, _ = tiny_data(np.random.default_rng(2))
    lm = data.drop_duplicates(subset=C.KEY)[C.KEY + ["treatment", "t_event_true"]]
    co, got = C.inject_censor_time(lm, data, 0.4, "i_administrative", 5, 80, 3.0)
    assert abs(got - 0.4) < 0.05
    assert co.shape == (len(lm),) and np.isfinite(co).any()
    co2, got2 = C.inject_censor_time(lm, data, 0.4, "i_administrative", 5, 80, 3.0)
    assert np.array_equal(co, co2) and got == got2


def test_cohort_and_corrected_offset():
    data, _ = tiny_data(np.random.default_rng(3))
    keep, counts = C.declared_cohort(data, {f"{GROUPS[0]}:1"})
    assert counts["tracks_removed"] == 1 and not keep["track_id"].eq(f"{GROUPS[0]}:1").any()
    lm = data.drop_duplicates(subset=C.KEY).assign(censor_offset=np.inf)
    off = C.corrected_censor_offset(lm, 10)
    none = lm["true_event"].eq("NONE").to_numpy()
    assert np.all(off[none] == 10 * np.floor((199 - lm["t_obs_frame"].to_numpy()[none]) / 10))
    assert np.all(np.isinf(off[~none]))
    assert C.reach_at_rate0(lm, 10, 8)["landmarks"] > 0


@pytest.mark.parametrize("regime", ["c_censoring_aware", "d_joint_survival"])
def test_run_unit_scores_every_held_out_row(regime, monkeypatch):
    from deadahead.survival import head
    monkeypatch.setitem(C.LGB_BINARY, "num_threads", 1)          # restored after the test
    monkeypatch.setitem(head.LGB_MULTI, "num_threads", 1)
    data, cols = tiny_data(np.random.default_rng(4))
    fold = {"fold": 0, "test_group": GROUPS[0]}
    p = C.run_unit(data, cols, fold, regime, 11, CFG, CFG_W)
    te = data[data["group"] == GROUPS[0]]
    for h in (30, 240):
        y = p[f"y_{h}"]
        assert len(y) == int((te["horizon_min"] == h).sum())
        rows = E.unit_scores(p, regime, lambda yt, s: float(np.mean(s[yt == 1])))
        assert {r["horizon_min"] for r in rows} >= {30}
    if regime == "d_joint_survival":
        assert p["correction"]["training_landmarks"] > 0
        assert p["d_mask_30"].sum() == len(p["y_30"])
    again = C.run_unit(data, cols, fold, regime, 11, CFG, CFG_W)
    assert np.array_equal(again[f"p_{regime}_LYTIC_30"], p[f"p_{regime}_LYTIC_30"])
