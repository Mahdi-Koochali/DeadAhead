"""Where the data lives."""

from deadahead import paths


def test_data_root_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DEADAHEAD_DATA", str(tmp_path))
    assert paths.data_root() == tmp_path.resolve()
    assert paths.raw_dir() == tmp_path.resolve() / "raw" / "celldeathdetect"
    assert paths.tables_dir() == tmp_path.resolve() / "tables"
    assert paths.crops_dir() == tmp_path.resolve() / "crops"
    assert paths.runs_dir() == tmp_path.resolve() / "runs"
    assert paths.bundle_dir() == tmp_path.resolve() / "bundle"


def test_unset_or_empty_data_root_is_the_checkout(monkeypatch):
    monkeypatch.delenv("DEADAHEAD_DATA", raising=False)
    assert paths.data_root() == (paths.REPO_ROOT / "data").resolve()
    monkeypatch.setenv("DEADAHEAD_DATA", "")
    assert paths.data_root() == (paths.REPO_ROOT / "data").resolve()
