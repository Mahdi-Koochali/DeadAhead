import sys

import pytest

from deadahead import __version__, cli


def test_every_command_prints_help_under_its_own_name(capsys):
    """Every command answers --help (DeepHit included, without PyTorch) and names itself in the usage line."""
    before = list(sys.argv)
    for name in cli.COMMANDS:
        with pytest.raises(SystemExit) as e:
            cli.main([name, "--help"])
        assert e.value.code == 0, name
        assert capsys.readouterr().out.startswith(f"usage: deadahead {name}"), name
        assert sys.argv == before


def test_unknown_command():
    assert cli.main(["no-such-command"]) == 2


def test_version(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out == f"deadahead {__version__}\n"


def test_missing_bundle_says_where_to_get_it(tmp_path, capsys):
    assert cli.main(["reproduce", "--bundle", str(tmp_path / "bundle")]) == 3
    err = capsys.readouterr().err
    assert "no MANIFEST.json" in err and "deadahead_bundle_v1.zip" in err and str(tmp_path) in err
