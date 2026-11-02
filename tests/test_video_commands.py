"""Argument checks of the video commands that run before anything heavy is read (no torch needed)."""
import pytest

from deadahead.video import clips, crops, features, selection as select, test_time
from deadahead.video.model import InputError, check_weights_name

NAME = "videomae_large_finetuned__DU145_bp__seed2601562.safetensors"


def test_weights_name_must_match_the_fold_and_seed():
    check_weights_name(NAME, "DU145_bp", 2601562)
    check_weights_name(f"somewhere/{NAME}", "DU145_bp", 2601562)
    check_weights_name("my_own_weights.safetensors", "PNT1A_st", 7)
    with pytest.raises(InputError, match="DU145_bp"):
        check_weights_name(NAME, "PNT1A_st", 2601562)
    with pytest.raises(InputError, match="2601562"):
        check_weights_name(NAME, "DU145_bp", 2601563)


@pytest.mark.parametrize("build", [crops.main, lambda argv: test_time.main(["masks", *argv])], ids=["crops", "masks"])
def test_an_unknown_movie_id_is_refused(build, tmp_path, monkeypatch, capsys):
    """One known and one mistyped id: the command stops before it builds anything."""
    monkeypatch.setenv("DEADAHEAD_DATA", str(tmp_path))
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "movies.csv").write_text("movie_id,image_path\nDU145_bp_1,a.tif\n")
    (tables / "cells.parquet").write_text("")
    out = tmp_path / "out"
    assert build(["--movies", "DU145_bp_1", "DU145_bp_9", "--out-dir", str(out)]) == 2
    assert "DU145_bp_9" in capsys.readouterr().err and not out.exists()


def test_crops_refuse_another_stride_in_the_all_frames_store(capsys):
    with pytest.raises(SystemExit) as e:
        crops.main(["--stride", "5"])
    assert e.value.code == 2 and "--out-dir" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["score", "--backbone-lr", "2e-4", "--layer-decay", "0.9"], ["choose"]])
def test_select_refuses_an_unknown_group(args, capsys):
    with pytest.raises(SystemExit) as e:
        select.main(args[:1] + ["--fold", "NOPE"] + args[1:])
    assert e.value.code == 2 and "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("args, message", [
    (["--seed", "1", "--adapted"], "--adapted takes no"),
    (["--weights", "w.safetensors"], "--seed"),
    (["--seed", "1"], "--weights"),
])
def test_features_arguments(args, message, capsys):
    with pytest.raises(SystemExit) as e:
        features.main(["--fold", "DU145_bp"] + args)
    assert e.value.code == 2 and message in capsys.readouterr().err


def test_the_first_commands_name_what_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DEADAHEAD_DATA", str(tmp_path))
    assert crops.main([]) == 3
    assert "deadahead data-inventory" in capsys.readouterr().err
    assert clips.main([]) == 3
    assert "deadahead data-windows" in capsys.readouterr().err
    assert test_time.main(["masks"]) == 3
    assert "deadahead data-inventory" in capsys.readouterr().err
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "movies.csv").write_text("movie_id\n")
    assert crops.main([]) == 3
    assert "deadahead data-cells" in capsys.readouterr().err
