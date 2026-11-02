import numpy as np
import pandas as pd

from deadahead.evaluation.aggregate import AHEAD_X
from deadahead.evaluation.rows import GROUPS, SEEDS
from deadahead.reproduce.decomposition import CONDITIONS, format_decomposition, part_decomposition
from deadahead.reproduce.main import available_parts
from test_table1 import predictions, synthetic_events, synthetic_rows


def bundle(seed: int = 0) -> dict:
    from deadahead.evaluation.rows import untyped_tracks
    rng = np.random.default_rng(seed)
    rows = synthetic_rows(rng)
    events = synthetic_events(rows)
    perfect = predictions(rows, rng, perfect=True)
    noise = predictions(rows, rng, perfect=False)
    determined = rows[rows["label"] != "CENSORED_IN_HORIZON"]          # the video model's rows
    video = predictions(determined, rng, perfect=True)
    blocks = []
    for x in CONDITIONS:
        b = predictions(determined, rng, perfect=False)
        b.insert(0, "condition", x)
        blocks.append(b)
    oracle = determined.assign(label=determined["label"].map({"APOPTOSIS": 1, "LYTIC": 2}).fillna(0).astype(int))
    return {"rows": rows, "events": events, "untyped": untyped_tracks(events),
            "oracle_labels": oracle[["group", "movie_id", "track_id", "t_obs_frame", "label"]],
            "predictions": {"deadahead": perfect, "videomae_finetuned": video},
            "adapted_predictions": noise, "test_time_predictions": pd.concat(blocks, ignore_index=True)}


def test_perfect_side_ahead_and_changes_fall():
    b = bundle()
    res = part_decomposition(b)
    assert res["population"]["n_rows"] == res["population"]["counts"]["cohort_rows"]
    for c in ("LYTIC", "APOPTOSIS"):
        r = res["fine_tuning"]["readings"][c]
        assert r["x"] == "deadahead" and r["y"] == "videomae_adapted_survival"
        assert r["x_values"]["nine_folds"]["macro"] == 1.0
        assert r["paired"]["nine_folds"]["outcome"] == AHEAD_X
        assert r["paired"]["nine_folds"]["x_ahead"] == len(GROUPS)
        t = res["test_time"]["readings"][c]
        assert t["intact"]["nine_folds"]["macro"] == 1.0
        for x in CONDITIONS:
            v = t["changed"][x]
            assert v["n_groups_where_it_fell"] == len(v["groups_where_it_fell"]) == len(GROUPS)
            assert v["intact_minus_changed"]["nine_folds"]["macro_difference"] > 0
        o = res["test_time_original_rows"][c]
        assert o["macro_intact"] == 1.0
        for x in CONDITIONS:
            v = o["conditions"][x]
            assert v["intact_minus_changed"]["k_plus"] + v["intact_minus_changed"]["k_minus"] \
                + v["intact_minus_changed"]["ties"] == len(GROUPS)
            assert v["macro_auprc"] < o["macro_intact"]
        q = res["fine_tuning_oracle_landmarks"][c]
        assert q["x_values"]["macro"] == 1.0 and q["paired"]["x_ahead"] == len(GROUPS)
        assert set(q["x_values"]["per_seed_macro"]) == set(SEEDS)
    text = format_decomposition(res)
    assert "tracked cell hidden" in text and "+" in text


def test_oracle_reading_needs_full_cover():
    import pytest
    b = bundle()
    b["oracle_labels"] = pd.concat([b["oracle_labels"], b["oracle_labels"].head(1).assign(t_obs_frame=-10)])
    with pytest.raises(ValueError):
        part_decomposition(b)
    b.pop("oracle_labels")
    assert "fine_tuning_oracle_landmarks" not in part_decomposition(b)


def test_part_available_only_with_its_files():
    b = bundle()
    assert "decomposition" in available_parts({**b, "movies": None, "inner_predictions": None})
    b.pop("adapted_predictions")
    assert "decomposition" not in available_parts(b)
