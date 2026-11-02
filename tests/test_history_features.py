"""The handcrafted history features on synthetic tracks."""

import numpy as np
import pandas as pd
import pytest

from deadahead.features.history import (
    BASE_FEATURES,
    HISTORY_COLUMNS,
    STATISTICS,
    build_history_features,
    history_columns,
)


def synthetic_cells(seed=0, lengths=(70, 46, 30), start=(0, 12, 5)):
    rng = np.random.default_rng(seed)
    parts = []
    for i, (n, s) in enumerate(zip(lengths, start)):
        d = {"track_id": f"m_1:{i + 1}", "frame_index": np.arange(s, s + n)}
        for f in BASE_FEATURES:
            d[f] = rng.normal(10.0, 2.0, size=n)
        parts.append(pd.DataFrame(d))
    cells = pd.concat(parts, ignore_index=True)
    cells.loc[3, "area_um2"] = 0.0          # a zero level: relchange must be NaN there, not inf
    return cells


def test_112_columns_in_order():
    hist = build_history_features(synthetic_cells(), 40)
    assert len(BASE_FEATURES) == 14 and len(STATISTICS) == 8
    assert history_columns(hist) == HISTORY_COLUMNS
    assert len(HISTORY_COLUMNS) == 14 * 8 == 112
    assert list(hist.columns[:2]) == ["track_id", "frame_index"]
    assert len(hist) == len(synthetic_cells())


@pytest.mark.parametrize("W", [5, 40])
@pytest.mark.parametrize("cut", [20, 45, 60])
def test_no_future_frames(W, cut):
    cells = synthetic_cells()
    ref = build_history_features(cells, W)
    later = cells["frame_index"] > cut
    # change every later frame, and separately remove them: rows at or before the cut must not move
    changed = cells.copy()
    rng = np.random.default_rng(1)
    for f in BASE_FEATURES:
        changed.loc[later, f] = rng.normal(50.0, 9.0, size=int(later.sum()))
    for other in (changed, cells[~later]):
        got = build_history_features(other, W)
        a = ref[ref["frame_index"] <= cut].reset_index(drop=True)
        b = got[got["frame_index"] <= cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_window_statistics_by_hand():
    W = 5
    cells = synthetic_cells()
    hist = build_history_features(cells, W).set_index(["track_id", "frame_index"])
    tr = cells[cells["track_id"] == "m_1:1"].set_index("frame_index")
    x = tr["dry_mass_pg"]
    t = 30
    win = x.loc[t - W:t]
    row = hist.loc[("m_1:1", t)]
    assert row["dry_mass_pg__last"] == x.loc[t]
    assert np.isclose(row["dry_mass_pg__mean"], win.mean(), rtol=0, atol=1e-12)
    assert np.isclose(row["dry_mass_pg__std"], win.std(ddof=1), rtol=0, atol=1e-12)
    assert row["dry_mass_pg__min"] == win.min() and row["dry_mass_pg__max"] == win.max()
    assert row["dry_mass_pg__delta"] == x.loc[t] - x.loc[t - W]
    assert row["dry_mass_pg__delta10"] == x.loc[t] - x.loc[t - 10]
    assert row["dry_mass_pg__relchange"] == (x.loc[t] - x.loc[t - W]) / x.loc[t - W]
    # the window needs W + 1 frames: NaN before, a value from frame W on
    assert np.isnan(hist.loc[("m_1:1", W - 1), "dry_mass_pg__mean"])
    assert np.isfinite(hist.loc[("m_1:1", W), "dry_mass_pg__mean"])
    assert not np.isinf(hist[[c for c in HISTORY_COLUMNS if c.endswith("__relchange")]].to_numpy()).any()


def test_input_order_and_other_tracks_do_not_matter():
    cells = synthetic_cells()
    ref = build_history_features(cells, 40)
    shuffled = cells.sample(frac=1.0, random_state=3)
    pd.testing.assert_frame_equal(ref, build_history_features(shuffled, 40), check_exact=True)
    one = cells[cells["track_id"] == "m_1:2"]
    a = ref[ref["track_id"] == "m_1:2"].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, build_history_features(one, 40).reset_index(drop=True), check_exact=True)


def test_cached_table_equals_a_fresh_build(tmp_path, monkeypatch):
    from deadahead.features.history import history_features
    monkeypatch.setenv("DEADAHEAD_DATA", str(tmp_path))
    (tmp_path / "tables").mkdir()
    cells = synthetic_cells()
    cells.to_parquet(tmp_path / "tables" / "cells.parquet", index=False)
    first = history_features(5)
    assert len(list((tmp_path / "runs" / "cache").glob("history_W5_*.parquet"))) == 1
    second = history_features(5)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    pd.testing.assert_frame_equal(first, build_history_features(cells, 5), check_exact=True)


def cached_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("DEADAHEAD_DATA", str(tmp_path))
    (tmp_path / "tables").mkdir()
    synthetic_cells().to_parquet(tmp_path / "tables" / "cells.parquet", index=False)
    return tmp_path / "runs" / "cache"


def test_parallel_writers_of_one_cache_do_not_collide(tmp_path, monkeypatch):
    """A writer that finishes while another is still writing must not take the other's temporary file."""
    from deadahead.features import history
    cache = cached_setup(tmp_path, monkeypatch)
    write, finished = pd.DataFrame.to_parquet, []

    def write_then_let_the_other_writer_finish(self, path, **kw):
        write(self, path, **kw)
        if not finished:
            finished.append(True)
            history.history_features(5)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", write_then_let_the_other_writer_finish)
    history.history_features(5)
    assert [p.suffix for p in cache.iterdir()] == [".parquet"]


def test_failed_cache_write_still_returns_the_table(tmp_path, monkeypatch, capsys):
    from deadahead.features import history
    cache = cached_setup(tmp_path, monkeypatch)
    fresh = build_history_features(synthetic_cells(), 5)
    write = pd.DataFrame.to_parquet

    def partial_write_then_disk_full(self, path, **kw):
        write(self, path, **kw)
        raise OSError("no space left on device")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", partial_write_then_disk_full)
    pd.testing.assert_frame_equal(history.history_features(5), fresh, check_exact=True)
    assert "cache not written" in capsys.readouterr().err
    assert list(cache.iterdir()) == []


def test_missing_cells_table_is_reported(tmp_path, monkeypatch, capsys):
    from deadahead.features import history
    monkeypatch.setenv("DEADAHEAD_DATA", str(tmp_path))
    assert history.main([]) == 3
    assert "run data-cells first" in capsys.readouterr().err


def test_history_window_constant_agrees_with_the_config():
    from deadahead.features.history import HISTORY_FRAMES_W
    from deadahead.survival.config import windows_config
    assert int(windows_config()["landmark"]["history_frames_W"]) == HISTORY_FRAMES_W
