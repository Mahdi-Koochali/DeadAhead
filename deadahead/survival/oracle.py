"""Evaluation labels: the outcome of every (landmark, horizon), read from the complete movie."""

from __future__ import annotations

import numpy as np
import pandas as pd

from deadahead.survival.head import CAUSE_IX, CAUSES, NO_EVENT


def build_oracle(windows: pd.DataFrame, events: pd.DataFrame, movies: pd.DataFrame,
                 cfg_w: dict) -> pd.DataFrame:
    """One row per (landmark, horizon) whose outcome is known in the complete movie, with its label.

    ``t_end_true`` is the last frame at which the cell's status is known: its death frame if it dies of a typed cause,
    otherwise the last frame of the full movie. A horizon is determined when the death falls inside it or when the
    cell is known event-free through it. ``oracle_label``, the complete-movie label, is 0 (no event), 1 (apoptosis) or
    2 (lytic death). An untyped death is not an event here; the scoring leaves those cells out
    (``evaluation.rows.remove_untyped``).
    """
    interval = float(cfg_w["frame_interval_min"])
    n_full = dict(zip(movies["movie_id"], movies["num_frames"]))

    ev = events.set_index("track_id")
    lm = windows[["movie_id", "track_id", "t_obs_frame", "group", "cell_line", "treatment",
                  "window_weight"]].drop_duplicates(subset=["movie_id", "track_id", "t_obs_frame"])
    # the onset interval travels with the landmark, for the interval-censored variant of the head
    for _c in ("t_onset_lower", "t_onset_upper"):
        if _c in ev.columns:
            lm[_c] = lm["track_id"].map(ev[_c])

    etype = lm["track_id"].map(ev["event_type"])
    tev = lm["track_id"].map(ev["t_event"])
    last_full = lm["movie_id"].map(n_full) - 1

    dies = etype.isin(CAUSES) & tev.notna()
    lm = lm.assign(
        true_event=np.where(dies, etype, "NONE"),
        t_event_true=np.where(dies, tev, np.nan),
        # a cell that never dies has known status up to the last frame of the full movie
        t_end_true=np.where(dies, tev, last_full),
    )

    rows = []
    for h in cfg_w["horizons_min"]:
        hf = int(h / interval)
        end = lm["t_obs_frame"] + hf
        within = dies & (lm["t_obs_frame"] < lm["t_event_true"]) & (lm["t_event_true"] <= end)
        clean = (~within) & (lm["t_end_true"] >= end)
        det = within | clean
        lab = np.where(within, lm["true_event"].map(CAUSE_IX).fillna(0), NO_EVENT)
        r = lm.assign(horizon_min=h, oracle_label=lab.astype(int), oracle_determined=det.values,
                      horizon_end_frame=end)
        rows.append(r[r["oracle_determined"]])
    return pd.concat(rows, ignore_index=True)
