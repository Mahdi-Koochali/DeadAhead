"""What deadahead reproduce prints and stores: the values the paper quotes, input checks, repeatable JSON."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_table1 import bundle  # noqa: F401
from test_table1 import synthetic_events, synthetic_rows

from deadahead.evaluation.aggregate import AHEAD_X, NOT_IMPORTANT, outcome_text
from deadahead.evaluation.alarms import warning_summary
from deadahead.evaluation.rows import GROUPS, SEEDS, untyped_tracks
from deadahead.reproduce import main as reproduce_main
from deadahead.reproduce.bundle import load_bundle
from deadahead.reproduce.data import EXCLUDED_ROUTES, format_data, part_data
from deadahead.reproduce.horizons import format_horizons
from deadahead.reproduce.main import PARTS, available_parts, main
from deadahead.reproduce.table1 import (chance_level, elapsed_time_fit, format_elapsed_time_fit, format_table1,
                                        part_table1)

ROOT = Path(__file__).resolve().parents[1]


def _unit(rate: float) -> dict:
    return {"achieved": {"alarmed_reference_rows": 1, "reference_rows": 2, "exposure_cell_hours": 1.0,
                         "rate_per_cell_hour": rate},
            "sustained": {"30": {"n_eligible_events": 3, "n_alarmed": 1, "share": 1 / 3, "chance_lower": 0.1,
                                 "chance_upper": 0.1}},
            "median_warning": {"n_events_all_eight_bins": 1, "n_events_left_out": 0, "median_min": 30.0}}


def test_warning_summary_key_order_does_not_follow_the_input_order():
    forward = {(g, s): _unit(i / 100) for i, g in enumerate(GROUPS) for s in SEEDS}
    backward = dict(reversed(list(forward.items())))
    a, b = warning_summary(forward), warning_summary(backward)
    assert json.dumps(a["sustained"]) == json.dumps(b["sustained"])
    assert list(a["sustained"]["30"]["eligible_deaths_per_group"]) == sorted(GROUPS)


def test_the_json_repeats_byte_for_byte_under_another_hash_seed(bundle, tmp_path):  # noqa: F811
    outs = []
    python_path = os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH"))))
    for seed in ("1", "2"):
        out = tmp_path / f"w{seed}.json"
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": python_path, "PYTHONDONTWRITEBYTECODE": "1"}
        subprocess.run([sys.executable, "-B", "-m", "deadahead", "reproduce", "--bundle", str(bundle), "--part",
                        "warnings", "--out", str(out)], check=True, env=env, cwd=tmp_path, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]
    assert "seconds" not in json.loads(outs[0])["warnings"]


def test_the_json_is_on_disk_before_the_report_is_printed(bundle, monkeypatch):  # noqa: F811
    def no_report(res):
        raise RuntimeError("no report")

    monkeypatch.setattr(reproduce_main, "report", no_report)
    out = bundle / "before_the_report.json"
    with pytest.raises(RuntimeError, match="no report"):
        main(["--bundle", str(bundle), "--part", "table1", "--out", str(out)])
    assert "table1" in json.loads(out.read_text())


def test_no_module_shadows_the_standard_library():
    names = {p.stem for d in ("evaluation", "reproduce", "report") for p in (ROOT / "deadahead" / d).glob("*.py")}
    assert not names & set(sys.stdlib_module_names)


def test_the_entry_module_runs_from_its_own_folder():
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH")))),
           "PYTHONDONTWRITEBYTECODE": "1"}
    folder = ROOT / "deadahead" / "reproduce"
    r = subprocess.run([sys.executable, "-B", "main.py", "--help"], env=env, cwd=folder, capture_output=True, text=True)
    assert r.returncode == 0 and "--part" in r.stdout


def test_table1_chance_row_is_in_the_json_and_on_the_page(bundle, capsys):  # noqa: F811
    out = bundle / "t.json"
    assert main(["--bundle", str(bundle), "--part", "table1", "--out", str(out)]) == 0
    res = json.loads(out.read_text())["table1"]
    assert res["chance"]["LYTIC"]["nine_folds"] == pytest.approx(chance_level(res["cohort"], "LYTIC"))
    assert res["chance"]["LYTIC"]["eight_folds_without_PNT1A_bp"] == pytest.approx(
        chance_level(res["cohort"], "LYTIC", tuple(g for g in GROUPS if g != "PNT1A_bp")))
    assert "chance (prevalence)" in capsys.readouterr().out


def test_chance_level_needs_no_particular_method():
    units = {f"{g}|s0": {"P": 1, "Q": 1, "N": 2, "U": 5} for g in GROUPS}
    res = {"methods": {"elapsed_time": {"LYTIC": {"units": units}}}}
    assert chance_level(res, "LYTIC") == 0.25


def _weights_reverse_the_slope() -> pd.DataFrame:
    """Early deaths of cells with few rows carry a large weight, so the weighted slope falls with time; most rows are
    late and hold the late deaths, so the unweighted slope rises."""
    out = []
    for g in GROUPS:
        for t in range(40, 400, 10):
            out.append({"movie_id": g, "track_id": f"{g}:{t}", "t_obs_frame": t, "group": g, "horizon_min": 30,
                        "label": "LYTIC" if t >= 380 else "NO_EVENT", "window_weight": 1.0})
            out.append({"movie_id": g, "track_id": f"{g}:e{t}", "t_obs_frame": t, "group": g, "horizon_min": 30,
                        "label": "LYTIC" if t == 40 else "NO_EVENT", "window_weight": 400.0 if t == 40 else 1.0})
    return pd.DataFrame(out)


def test_the_elapsed_time_fit_shows_the_weights_reversing_the_slope():
    rows = _weights_reverse_the_slope()
    fit = elapsed_time_fit({"rows": rows}, rows)["LYTIC"]
    assert fit["weighted"]["groups_slope_below_zero"] == 9 and fit["unweighted"]["groups_slope_above_zero"] == 9
    assert fit["unweighted"]["auprc"]["nine_folds"]["macro"] > fit["weighted"]["auprc"]["nine_folds"]["macro"]


def test_the_weighted_fit_is_the_table1_entry(bundle):  # noqa: F811
    b = load_bundle(bundle)
    t1 = part_table1(b)
    for c in ("LYTIC", "APOPTOSIS"):
        assert t1["elapsed_time_fit"][c]["weighted"]["auprc"]["nine_folds"]["macro"] == pytest.approx(
            t1["cohort"]["methods"]["elapsed_time"][c]["nine_folds"]["macro"], abs=1e-12)
    text = format_table1(t1)
    assert "chance (prevalence)" in text and format_elapsed_time_fit(t1["elapsed_time_fit"]) in text


def test_format_horizons_prints_the_columns_of_the_paper():
    blk = {"nine_folds": {"macro": 0.5}}
    rate = {k: blk for k in ("prevalence", "achieved_rate_equal_group_macro", "precision", "sensitivity", "auprc")}
    est = {"ipa": blk, "ipa_training_null": blk, "c_no_same_track": blk,
           "ipcw_slope": {"nine_folds": {"macro": 0.634}}, "e_over_o": {"nine_folds": {"macro": 1.274}},
           "ipa_groups_below_zero": {"n_below_zero": 4}}
    res = {"survival_estimators": {"deadahead": {c: {h: est for h in ("30", "60", "120", "240")}
                                                 for c in ("LYTIC", "APOPTOSIS")}},
           "models": {"deadahead": {c: {h: {"0.2_per_cell_hour": rate} for h in ("30", "60", "120", "240")}
                                    for c in ("LYTIC", "APOPTOSIS")}}}
    text = format_horizons(res)
    assert "slope" in text and "P/O" in text
    row = next(x for x in text.splitlines() if x.startswith("lytic") and "deadahead" in x)
    assert "0.634" in row and "1.274" in row and row.split()[3] == "50.0"


def _events() -> pd.DataFrame:
    """Two movies, A scored and B left out; A holds one of each fate, one death after the cut, one excluded death."""
    rows = [("A:1", "APOPTOSIS", 100.0), ("A:2", "LYTIC", 200.0), ("A:3", "LYTIC", 500.0), ("A:4", "CENSORED", np.nan),
            ("A:5", "UNTYPED_DEATH", 100.0), ("A:6", "UNTYPED_DEATH", 600.0), ("A:7", "EXCLUDED", np.nan),
            ("B:1", "APOPTOSIS", 50.0)]
    return pd.DataFrame(rows, columns=["track_id", "event_type", "t_event"])


def test_part_data_counts_every_scope():
    ev = _events()
    rows = pd.DataFrame({"movie_id": "A", "track_id": ["A:1", "A:2", "A:4", "A:5"], "t_obs_frame": 40, "group": "g",
                         "label": ["APOPTOSIS", "NO_EVENT", "CENSORED_IN_HORIZON", "UNTYPED_DEATH"], "horizon_min": 30,
                         "window_weight": 1.0})
    b = {"events": ev, "rows": rows, "untyped": untyped_tracks(ev),
         "movies": pd.DataFrame({"movie_id": ["A", "B"], "num_frames": [439, 863]})}
    res = part_data(b)
    assert res["movies"]["left_out"] == ["B"] and res["movies"]["scored"] == 1
    a = res["cells_all_movies"]
    assert a["cells"] == 8 and a["by_event_type"]["APOPTOSIS"] == 2 and a["by_event_type"]["EXCLUDED"] == 1
    assert a["at_the_cut"]["apoptosis"] == 2 and a["at_the_cut"]["lytic"] == 1
    assert a["at_the_cut"]["censored"] == 2 and a["at_the_cut"]["untyped_before_the_cut"] == 1
    assert res["cells_scored_movies"]["by_event_type"]["APOPTOSIS"] == 1
    assert "not_computed" in res["route_totals"]
    f = res["forecasts"]
    assert (f["rows_scored_movies"], f["cohort_rows"], f["apoptosis_positives"]) == (4, 2, 1)
    assert "1 rows of the 1 untyped-death cells" in format_data(res)


def test_the_route_totals_add_the_excluded_deaths():
    from deadahead.reproduce.data import route_totals
    n = {"APOPTOSIS": 635, "LYTIC": 1117, "CENSORED": 378, "UNTYPED_DEATH": 12, "EXCLUDED": 13}
    r = route_totals(n)
    assert (r["apoptosis"], r["lytic"]) == (646, 1119) and sum(EXCLUDED_ROUTES.values()) == 13


@pytest.mark.parametrize("part", ["decomposition", "censoring", "gap"])
def test_a_part_without_its_inputs_stops_with_a_message(tmp_path, part, capsys):
    rows = synthetic_rows(np.random.default_rng(1))
    ev = synthetic_events(rows)
    rows.to_parquet(tmp_path / "rows.parquet", index=False)
    ev.to_parquet(tmp_path / "events.parquet", index=False)
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"files": {"rows.parquet": {}, "events.parquet": {}}}))
    assert main(["--bundle", str(tmp_path), "--part", part, "--out", str(tmp_path / "x.json")]) == 3
    assert "error: the bundle holds no" in capsys.readouterr().err


def test_the_gap_part_names_a_missing_movie_table():
    from deadahead.reproduce.gap import part_gap
    with pytest.raises(ValueError, match="movie table"):
        part_gap({"landmarks": object(), "cumulative_incidence": object()})


def test_every_part_is_known_to_available_parts():
    assert set(available_parts({"events": None})) <= set(PARTS)


def test_outcome_text_names_the_x_side():
    assert outcome_text({"outcome": AHEAD_X}) == "DeadAhead ahead (meets the decision rule)"
    assert outcome_text({"outcome": NOT_IMPORTANT}) == NOT_IMPORTANT


@pytest.mark.parametrize("name", ["plot-warnings", "plot-gap-window", "plot-alarm-rate"])
def test_figure_commands_refuse_a_missing_or_wrong_table(tmp_path, name, capsys):
    from deadahead import cli
    out = tmp_path / "new" / "f.png"
    assert cli.main([name, str(tmp_path / "none.csv"), str(out)]) == 3
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("a,b\n1,2\n")
    assert cli.main([name, str(wrong), str(out)]) == 3
    err = capsys.readouterr().err
    assert "no table at" in err and "is not this figure's table" in err and not out.exists()
