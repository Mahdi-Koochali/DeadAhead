"""The figures of the supplement: their tables from the reproduce output, the drawing and the plot-figures command, and
the tables read from the predictions bundle, on synthetic inputs."""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deadahead import cli
from deadahead.evaluation.alarms import threshold_and_apply
from deadahead.evaluation.rows import GROUPS, SEEDS
from deadahead.report import bundle_tables as B
from deadahead.report import figures as F
from deadahead.reproduce.image_warnings import IMAGE
from deadahead.reproduce.warning_times import DEADAHEAD

ROUTES = ("APOPTOSIS", "LYTIC")
H = ("30", "60", "120", "240")


def block(base: float) -> dict:
    per_group = {g: base + 0.01 * i for i, g in enumerate(GROUPS)}
    return {"macro": float(np.mean(list(per_group.values()))), "per_group": per_group}


def results() -> dict:
    """The parts of a reproduce output that the overview figures read."""
    methods = {key: {r: {"nine_folds": block(0.1 + 0.01 * i)} for r in ROUTES}
               for i, (key, _) in enumerate(F.TABLE1_ROWS)}
    differences = {g: 0.02 * (i - 3) for i, g in enumerate(GROUPS)}
    contrasts = {f"deadahead - dinov2_handcrafted|refit|{r}": {
        "nine_folds": {"per_group_difference": differences, "macro_difference": 0.02}} for r in ROUTES}
    table1 = {"cohort": {"methods": methods, "contrasts": contrasts},
              "chance": {r: {"nine_folds": 0.02} for r in ROUTES}}
    changed = {key: {"auprc": {"nine_folds": block(0.1)}} for key, _ in F.TEST_TIME_ROWS}
    readings = {r: {"intact": {"nine_folds": block(0.3)}, "changed": changed} for r in ROUTES}
    alarms = {r: {h: {"0.2_per_cell_hour": {"prevalence": {"nine_folds": {"macro": 0.01 * int(h) / 30}},
                                            "auprc": {"nine_folds": {"macro": 0.3 + int(h) / 1000}}}}
                  for h in H} for r in ROUTES}
    survival = {r: {h: {"c_no_same_track": {"nine_folds": {"macro": 0.9 - int(h) / 1000}},
                        "ipa": {"nine_folds": {"macro": 0.03 - int(h) / 1000}}} for h in H} for r in ROUTES}
    return {"table1": table1, "decomposition": {"test_time": {"readings": readings}},
            "horizons": {"models": {"deadahead": alarms}, "survival_estimators": {"deadahead": survival}}}


def test_the_overview_tables_hold_one_row_per_group_and_the_macro():
    res = results()
    t = F.table1_table(res)
    assert set(t["route"]) == set(ROUTES)
    one = t[(t["route"] == "LYTIC") & (t["name"] == "DeadAhead (ours)")]
    assert len(one) == len(GROUPS) + 1
    macro = float(one.loc[one["group"] == "macro", "auprc"].iloc[0])
    assert macro == pytest.approx(one.loc[one["group"] != "macro", "auprc"].mean())
    assert float(t[(t["route"] == "LYTIC") & (t["name"] == "Chance")]["auprc"].iloc[0]) == 0.02
    assert not t["name"].str.contains(r"\|").any(), "a key of the reproduce output in a printed name"

    d = F.group_differences_table(res)
    assert len(d) == 2 * (len(GROUPS) + 1)
    assert set(d["group"]) == {g.replace("_", " ") for g in GROUPS} | {"macro"}

    h = F.horizons_table(res)
    assert sorted(h["horizon_min"].unique()) == [30, 60, 120, 240]
    assert list(h.columns) == ["route", "horizon_min", "prevalence", "auprc", "concordance", "ipa"]

    s = F.test_time_table(res)
    assert set(s["name"]) == {"Chance", "Intact", "Cell hidden", "Frames shuffled", "Last frame repeated"}


def test_every_figure_is_drawn_and_repeats_byte_for_byte(tmp_path):
    res = results()
    for name, build, plot in (("t1", F.table1_table, F.plot_table1), ("s1", F.horizons_table, F.plot_horizons),
                              ("s2", F.group_differences_table, F.plot_group_differences),
                              ("s4", F.test_time_table, F.plot_test_time)):
        a, b = tmp_path / f"{name}_a.png", tmp_path / f"{name}_b.png"
        plot(build(res), a)
        plot(build(res), b)
        assert a.read_bytes() == b.read_bytes(), name


def test_the_axes_are_not_cut_above_their_baseline(tmp_path, monkeypatch):
    """AUPRC starts at 0 and concordance at chance, so no figure enlarges a difference."""
    seen = {}
    real = F._save

    def spy(fig, out):
        seen[Path(out).stem] = [ax.get_ylim() for ax in fig.axes]
        real(fig, out)

    monkeypatch.setattr(F, "_save", spy)
    F.plot_horizons(F.horizons_table(results()), tmp_path / "s1.png")
    (auprc, concordance, _) = seen["s1"]
    assert auprc[0] == 0 and concordance == (0.5, 1.0)


def test_plot_figures_writes_each_figure_with_its_table(tmp_path):
    drawn = F.draw_from_results(results(), tmp_path)
    assert set(drawn) == {F.TABLE1_FIG, F.HORIZONS_FIG, F.GROUPS_FIG, F.TEST_TIME_FIG}
    for stem in drawn:
        assert (tmp_path / f"{stem}.png").stat().st_size > 1000
        assert len(pd.read_csv(tmp_path / "data" / f"{stem}.csv")) > 0


def test_the_command_refuses_a_missing_output(tmp_path, capsys):
    assert cli.main(["plot-figures", str(tmp_path / "none.json"), "--no-bundle"]) == 3
    assert "no reproduce output" in capsys.readouterr().err


def test_the_command_refuses_an_output_without_any_part(tmp_path, capsys):
    p = tmp_path / "all.json"
    p.write_text("{}")
    assert cli.main(["plot-figures", str(p), "--no-bundle", "--out-dir", str(tmp_path / "o")]) == 3
    assert "no part" in capsys.readouterr().err


def test_the_command_refuses_a_bundle_without_the_warning_inputs(tmp_path, capsys, monkeypatch):
    import deadahead.reproduce.bundle as bundle
    (tmp_path / "MANIFEST.json").write_text("{}")
    results_file = tmp_path / "all.json"
    results_file.write_text("{}")
    monkeypatch.setattr(bundle, "load_bundle", lambda d, verify=False: {"rows": None})
    assert cli.main(["plot-figures", str(results_file), "--bundle", str(tmp_path), "--out-dir", str(tmp_path / "o")]) == 3
    assert "holds no movies" in capsys.readouterr().err


def inputs() -> dict:
    """One death by each route per group, 8 forecasts before it, and 40 reference forecasts; the scores rise to the
    death."""
    rows, scores = {}, {DEADAHEAD: {}, IMAGE: {}}
    for g in GROUPS:
        lead = np.r_[np.arange(30.0, 241.0, 30.0) - 15, np.arange(30.0, 241.0, 30.0) - 15, np.full(40, np.nan)]
        cause = np.r_[np.full(8, 2), np.full(8, 1), np.zeros(40, int)]
        rows[g] = pd.DataFrame({"lead_min": lead, "event_cause": cause, "negative": cause == 0,
                                "track_id": ["a"] * 8 + ["b"] * 8 + [f"r{i}" for i in range(40)]})
        te = np.r_[np.linspace(1, 0.5, 8), np.linspace(1, 0.5, 8), np.linspace(0, 0.4, 40)]
        ref = np.linspace(0, 0.4, 200)
        for s in SEEDS:
            for c in ROUTES:
                scores[DEADAHEAD][(g, s, c)] = (ref, te)
                scores[IMAGE][(g, s, c)] = (ref, te * 0.8)
    return {"rows": rows, "scores": scores}


def test_the_alarm_share_counts_the_forecasts_of_each_bin_and_the_reference():
    t = B.alarm_share_table(inputs())
    assert len(t) == 2 * B.BINS
    assert set(t["n_groups"]) == {len(GROUPS)}
    # every forecast of a dying cell scores above the whole reference, so every alarm is on
    assert (t["alarm_share_pct"] == 100).all()
    ref, te = inputs()["scores"][DEADAHEAD][(GROUPS[0], SEEDS[0], "LYTIC")]
    flags, _ = threshold_and_apply(ref, te, Fraction(1, 10))
    neg = inputs()["rows"][GROUPS[0]]["negative"].to_numpy(bool)
    assert t["reference_alarm_share_pct"].iloc[0] == pytest.approx(100 * flags[neg].mean())


def test_the_match_group_by_group_raises_as_many_reference_alarms():
    t = B.group_match_table(inputs())
    assert list(t["group"])[-1] == "macro"
    macro = t.iloc[-1]
    assert macro["deadahead_warned_pct"] == pytest.approx(t["deadahead_warned_pct"].iloc[:-1].mean())
    assert macro["n_groups"] == len(GROUPS)
    difference = macro["deadahead_warned_pct"] - macro["dinov2_handcrafted_warned_pct"]
    assert macro["difference_points"] == pytest.approx(difference)


def test_the_row_weights_are_shares_of_the_determined_rows():
    rows = pd.DataFrame({"horizon_min": 30, "label": ["LYTIC", "NO_EVENT", "NO_EVENT", "CENSORED_IN_HORIZON"],
                         "window_weight": [3.0, 1.0, 2.0, 100.0]})
    t = B.row_weight_table({"rows": rows}).set_index("label")
    assert "CENSORED_IN_HORIZON" not in t.index
    assert t.loc["LYTIC", "share_of_rows_pct"] == pytest.approx(100 / 3)
    assert t.loc["LYTIC", "share_of_weight_pct"] == pytest.approx(50)
