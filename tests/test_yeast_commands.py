"""The yeast commands from the command line on synthetic data: yeast-data on a small archive, yeast-score, yeast-rules
and yeast-survival, the bundle check, incomplete bundles and bad input."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from deadahead.paths import REPO_ROOT
from deadahead.reproduce.bundle import sha256_file
from deadahead.yeast import constants as C
from deadahead.yeast import data as D
from deadahead.yeast import finetune as FT
from deadahead.yeast import lifespans as L
from deadahead.yeast import predict as PR
from deadahead.yeast import rules as B
from deadahead.yeast import score as S
from deadahead.yeast import survival as SV

pytest.importorskip("lightgbm")

from test_yeast_score import synthetic_series, write_bundle  # noqa: E402


def small_archive(path, series) -> None:
    """A zip laid out as the DetecDiv archive: `deep` (the classes) and `vid` (60 x 59 x 3 x T) per trap."""
    from scipy.io import savemat
    rng = np.random.default_rng(2)
    with zipfile.ZipFile(path, "w") as z:
        for trap, cls in series.items():
            vid = rng.integers(0, 256, size=(*C.CROP_SHAPE, len(cls)), dtype=np.uint8)
            buf = io.BytesIO()
            savemat(buf, {"deep": cls.astype(np.uint8)[None, :], "vid": vid})
            z.writestr(f"lifespanAnnotation/timeseries/lstm_labeled_{trap}.mat", buf.getvalue())


def counts_of(series) -> dict:
    rows, notes = D.build_rows(series)
    return {"TRAPS_BY_POSITION": {p: sum(D.trap_position(t) == p for t in series) for p in C.POSITIONS},
            "DEATHS_BY_POSITION": {p: int(rows.loc[(rows["position"] == p) & rows["has_event"], "unit"].nunique())
                                   for p in C.POSITIONS},
            "N_LIFESPANS": notes["n_lifespans"], "N_LANDMARKS": len(rows)}


@pytest.fixture()
def archive(tmp_path, monkeypatch):
    """A small archive with the constants set to its counts."""
    series = synthetic_series(n_traps=2, n_frames=80)
    path = tmp_path / "lifespanAnnotation.zip"
    small_archive(path, series)
    for name, value in counts_of(series).items():
        monkeypatch.setattr(C, name, value)
    return path, series


def write_manifest(bundle) -> None:
    files = {str(p.relative_to(bundle)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
             for p in sorted(bundle.rglob("*")) if p.is_file()}
    (bundle / "MANIFEST.json").write_text(json.dumps({"files": files}))


def test_published_counts_add_up():
    assert sum(C.TRAPS_BY_POSITION.values()) == 250 and C.N_LIFESPANS == 249
    assert sum(C.DEATHS_BY_POSITION.values()) == 173


def test_check_counts_on_altered_counts(archive):
    _, series = archive
    rows, notes = D.build_rows(series)
    D.check_counts(rows, notes, series)
    fewer = {k: v for k, v in series.items() if k != f"{C.POSITIONS[1]}_1"}
    with pytest.raises(D.StoreError, match="traps per position"):
        D.check_counts(rows, notes, fewer)
    alive = {k: np.where(v == L.DEAD, L.CLOG, v) if k == f"{C.POSITIONS[2]}_2" else v for k, v in series.items()}
    with pytest.raises(D.StoreError, match="deaths per position"):
        D.check_counts(*D.build_rows(alive), alive)
    with pytest.raises(D.StoreError, match="lifespans, expected"):
        D.check_counts(rows, {**notes, "n_lifespans": notes["n_lifespans"] + 1}, series)
    with pytest.raises(D.StoreError, match="landmarks, expected"):
        D.check_counts(rows.iloc[:-1], notes, series)


def test_yeast_data_on_a_small_archive(tmp_path, archive):
    path, series = archive
    out = tmp_path / "yeast"
    assert D.main(["--zip", str(path), "--out", str(out)]) == 0
    assert D.tables_missing(out) == []
    got, rows, folds, pics = D.load_tables(out)
    assert sorted(got) == sorted(series) and np.array_equal(got[f"{C.POSITIONS[0]}_1"], series[f"{C.POSITIONS[0]}_1"])
    assert len(rows) == C.N_LANDMARKS and set(pics) == set(C.POSITIONS)
    fold = folds["folds"][C.POSITIONS[0]]
    assert fold["channel"] in (0, 1, 2) and 0 < fold["m_D"] < 1 and fold["size_rule"]["side"] in ("below", "above")
    assert folds["source_moments"] == pytest.approx(D.config_source_moments())
    assert np.load(out / "vid_store" / f"{C.POSITIONS[0]}_1.npy").shape == (80, *C.CROP_SHAPE)
    rec = json.loads((out / "rows" / "row_files.json").read_text())
    p = C.POSITIONS[3]
    assert rec["folds"][p]["final"]["rows"] == len(pd.read_parquet(out / "rows" / f"final__{p}.parquet"))
    assert p not in set(D.read_row_file(out / "rows", "final", p)["position"])


def test_yeast_data_refuses_a_wrong_archive_before_writing(tmp_path, capsys):
    path = tmp_path / "other.zip"
    small_archive(path, synthetic_series(n_traps=2, n_frames=80))
    out = tmp_path / "yeast"
    assert D.main(["--zip", str(path), "--out", str(out)]) == 3
    assert "not the DetecDiv lifespan archive" in capsys.readouterr().err and not out.exists()
    not_zip = tmp_path / "a.zip"
    not_zip.write_text("no archive")
    assert D.main(["--zip", str(not_zip), "--out", str(out)]) == 3
    assert D.main(["--zip", str(tmp_path / "none.zip"), "--out", str(out)]) == 3
    with zipfile.ZipFile(tmp_path / "empty.zip", "w") as z:
        z.writestr("readme.txt", "no timeseries")
    assert D.main(["--zip", str(tmp_path / "empty.zip"), "--out", str(out)]) == 3


def test_yeast_data_labels_only(tmp_path, archive):
    out = tmp_path / "yeast"
    assert D.main(["--zip", str(archive[0]), "--out", str(out), "--no-pixels"]) == 0
    assert D.tables_missing(out) == ["brightness.parquet", "size.parquet", "folds.json"]
    assert not (out / "vid_store").exists() and (out / "rows" / "all_keys.parquet").is_file()


def test_yeast_score_command(tmp_path, fast_rules, capsys):
    bundle, _ = write_bundle(tmp_path, synthetic_series(n_traps=3, n_frames=200))
    out = tmp_path / "scores.json"
    args = ["--bundle", str(bundle), "--no-sensitivity", "--no-lead-time", "--out", str(out)]
    assert S.main([*args, "--no-verify"]) == 0
    text = capsys.readouterr().out
    assert "fine-tuned start against the best rule" in text and "comparison_A" not in text
    rec = json.loads(out.read_text())
    assert set(rec["summary"]["macro_auprc_30min"]) == set(C.STARTS)


def test_yeast_score_checks_the_bundle_against_its_manifest(tmp_path, fast_rules, capsys):
    bundle, _ = write_bundle(tmp_path, synthetic_series(n_traps=3, n_frames=200))
    args = ["--bundle", str(bundle), "--no-sensitivity", "--no-lead-time"]
    assert S.main(args) == 3                                       # no MANIFEST.json
    assert "--no-verify" in capsys.readouterr().err
    write_manifest(bundle)
    assert S.main(args) == 0
    capsys.readouterr()
    victim = bundle / "predictions" / f"{S.run_name(C.STARTS[0], C.POSITIONS[0], C.SEEDS[0])}.parquet"
    victim.write_bytes(victim.read_bytes()[:-1] + b"x")
    assert S.main(args) == 3
    assert "does not match its manifest" in capsys.readouterr().err
    victim.unlink()
    assert S.main(args) == 3
    assert "missing" in capsys.readouterr().err


def test_yeast_score_no_bundle_and_no_folder(tmp_path, capsys):
    assert S.main(["--bundle", str(tmp_path / "none")]) == 3
    assert "no yeast bundle" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        S.main(["--bundle", str(tmp_path / "none"), "--out", str(tmp_path / "no" / "scores.json")])
    assert e.value.code == 2


def test_yeast_score_two_seeds_are_not_scorable(tmp_path, fast_rules, capsys):
    bundle, _ = write_bundle(tmp_path, synthetic_series(n_traps=3, n_frames=200), n_seeds=2)
    assert S.main(["--bundle", str(bundle), "--no-verify", "--no-sensitivity", "--no-lead-time"]) == 0
    text = capsys.readouterr().out
    assert "not scorable" in text and "None" not in text
    rec = S.score_bundle(bundle, with_sensitivity=False, with_lead_time=False, verify=False)
    assert rec["comparison_A"] == "not scorable" and rec["summary"]["ahead_of_best_rule"] == {}
    assert set(rec["summary"]["macro_auprc_30min"]) == set(C.STARTS)


def test_yeast_score_one_start_absent(tmp_path, fast_rules, capsys):
    bundle, _ = write_bundle(tmp_path, synthetic_series(n_traps=3, n_frames=200))
    for f in (bundle / "predictions").glob(f"{C.STARTS[1]}__*"):
        f.unlink()
    assert S.main(["--bundle", str(bundle), "--no-verify", "--no-lead-time"]) == 0
    text = capsys.readouterr().out
    assert "not scorable" in text and "None" not in text
    rec = S.score_bundle(bundle, with_sensitivity=False, with_lead_time=False, verify=False)
    assert set(rec["summary"]["macro_auprc_30min"]) == {C.STARTS[0]}
    assert rec["comparison_B"] == "not scorable: needs both starts"
    assert rec["summary"]["finetuned_ahead_of_public_start"] is None


def test_yeast_score_one_position_absent(tmp_path, fast_rules):
    bundle, _ = write_bundle(tmp_path, synthetic_series(n_traps=3, n_frames=200))
    for f in (bundle / "predictions").glob(f"{C.STARTS[0]}__{C.POSITIONS[3]}__*"):
        f.unlink()
    rec = S.score_bundle(bundle, with_sensitivity=False, with_lead_time=False, verify=False)
    assert rec["scorable"][C.STARTS[0]]["scorable"] is False
    assert set(rec["summary"]["macro_auprc_30min"]) == {C.STARTS[1]}


def test_yeast_score_a_position_without_a_death(tmp_path, fast_rules, capsys):
    series = synthetic_series(n_traps=3, n_frames=200)
    p = C.POSITIONS[2]
    calm = {k: (np.where(v == L.DEAD, L.CLOG, v) if D.trap_position(k) == p else v) for k, v in series.items()}
    bundle, _ = write_bundle(tmp_path, calm)
    assert S.main(["--bundle", str(bundle), "--no-verify", "--no-sensitivity", "--no-lead-time"]) == 3
    assert p in capsys.readouterr().err


def test_yeast_rules_command(tmp_path, fast_rules, capsys):
    bundle, _ = write_bundle(tmp_path, synthetic_series(n_traps=3, n_frames=200))
    out = tmp_path / "rules.json"
    assert B.main(["--tables", str(bundle / "labels"), "--horizons", "30", "--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert set(rec) == set(C.POSITIONS) and set(rec[C.POSITIONS[0]]) == {"30"}
    assert "bar" in rec[C.POSITIONS[0]]["30"]
    capsys.readouterr()
    assert B.main(["--tables", str(tmp_path / "none")]) == 3
    assert "yeast-data" in capsys.readouterr().err
    for bad in (["--horizons", "45"], ["--out", str(tmp_path / "no" / "rules.json")]):
        with pytest.raises(SystemExit) as e:
            B.main(["--tables", str(bundle / "labels"), *bad])
        assert e.value.code == 2


def test_yeast_survival_command(tmp_path, fast_rules, capsys):
    series = synthetic_series(n_traps=3, n_frames=200)
    rows, _ = D.build_rows(series)
    rows_dir, runs = tmp_path / "rows", tmp_path / "runs"
    D.write_row_files(rows, rows_dir)
    keys = rows[["unit", "position", "t"]].reset_index(drop=True)
    run = f"{C.STARTS[1]}__{C.POSITIONS[0]}__s{C.SEEDS[1]}"
    args = ["--run", run, "--rows-dir", str(rows_dir), "--runs-dir", str(runs)]
    assert SV.main(args) == 3                                      # no embeddings yet
    assert "yeast-finetune" in capsys.readouterr().err
    (runs / "embed").mkdir(parents=True)
    np.save(runs / "embed" / f"{run}.npy", np.random.default_rng(0).normal(size=(len(keys), 8)).astype(np.float32))
    keys.to_parquet(runs / "embed" / f"{run}.keys.parquet", index=False)
    assert SV.main(args) == 0
    held = keys[keys["position"] == C.POSITIONS[0]].reset_index(drop=True)
    for row, cols in (("S", ["cif_30"]), ("SL", ["cif_240"]), ("NB", ["cif_bud_30", "cif_death_240"])):
        fr = pd.read_parquet(runs / "surv" / f"{run}__{row}.parquet")
        assert fr[["unit", "t"]].equals(held[["unit", "t"]]) and set(cols) <= set(fr.columns)
        assert np.isfinite(fr[cols].to_numpy()).all()
    with pytest.raises(SystemExit) as e:
        SV.main(["--run", "le__videomae_public__Pos3_3"])
    assert e.value.code == 2


def test_commands_without_pytorch_name_the_install(tmp_path, monkeypatch, capsys):
    weights = tmp_path / "w.pt"
    weights.touch()
    monkeypatch.setitem(sys.modules, "torch", None)
    for main, args in ((FT.main, ["--start", "videomae_public", "--heldout", "Pos3_3"]),
                       (PR.main, ["--weights", str(weights), "--heldout", "Pos3_3"])):
        with pytest.raises(SystemExit) as e:
            main(args)
        assert e.value.code == 2 and 'pip install -e ".[video]"' in capsys.readouterr().err


def test_bad_arguments_stop_with_a_message(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        FT.main(["--start", "videomae_public", "--heldout", "Pos3_3", "--kind", "thr", "--seed", "1"])
    assert e.value.code == 2 and "use seed 2601562" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        PR.main(["--weights", str(tmp_path / "none.pt"), "--heldout", "Pos3_3"])
    assert e.value.code == 2 and "--weights" in capsys.readouterr().err


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="the platform already lacks it")
def test_rules_import_without_sched_getaffinity():
    code = ("import os\n"
            "del os.sched_getaffinity\n"
            "os.environ.pop('OMP_NUM_THREADS', None)\n"
            "from deadahead.yeast import rules\n"
            "print(rules.LGB_SURV['num_threads'] == (os.cpu_count() or 1))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "True", r.stderr


def test_load_tables_needs_every_file(tmp_path):
    bundle, _ = write_bundle(tmp_path, synthetic_series(n_traps=2, n_frames=80))
    labels = bundle / "labels"
    assert D.tables_missing(labels) == []
    (labels / "size.parquet").unlink()
    assert D.tables_missing(labels) == ["size.parquet"]
    shutil.rmtree(labels)
    assert D.tables_missing(labels) == list(D.TABLE_FILES)
