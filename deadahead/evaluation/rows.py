"""Forecast rows: row keys, score alignment, the evaluation cohort and the outcome sets P, Q, N, U."""

from __future__ import annotations

import numpy as np
import pandas as pd

#: the nine held-out groups (cell line x treatment), in fold order
GROUPS = ("DU145_bp", "DU145_do", "DU145_st", "LNCaP_bp", "LNCaP_do", "LNCaP_st", "PNT1A_bp", "PNT1A_do", "PNT1A_st")
SEEDS = (2601562, 2601563, 2601564)
CAUSES = ("APOPTOSIS", "LYTIC")
OTHER = {"APOPTOSIS": "LYTIC", "LYTIC": "APOPTOSIS"}
KEY = ["movie_id", "track_id", "t_obs_frame"]
TABLE1_HORIZON = 30
LABEL_UNTYPED = "UNTYPED_DEATH"
LABEL_CENSORED = "CENSORED_IN_HORIZON"
LABEL_EVENT_FREE = "NO_EVENT"
WINDOW_LABELS = ("APOPTOSIS", "LYTIC", LABEL_UNTYPED, LABEL_EVENT_FREE, LABEL_CENSORED)


def inner_groups(g: str) -> list[str]:
    """The eight training groups of held-out group g, in fold order."""
    if g not in GROUPS:
        raise ValueError(f"unknown outer group {g!r}")
    return [h for h in GROUPS if h != g]


# ------------------------------------------------------------------ row keys
def key_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"movie_id": df["movie_id"].astype(str).to_numpy(),
                         "track_id": df["track_id"].astype(str).to_numpy(),
                         "t_obs_frame": df["t_obs_frame"].astype(np.int64).to_numpy()})


def key_index(df: pd.DataFrame) -> pd.MultiIndex:
    k = key_frame(df)
    return pd.MultiIndex.from_arrays([k["movie_id"].to_numpy(), k["track_id"].to_numpy(), k["t_obs_frame"].to_numpy()])


def align_scores(file_keys: pd.DataFrame, scores: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    """The file's scores at ``rows``' keys; raises when a row has no prediction or a file key is duplicated."""
    fi = key_index(file_keys)
    if not fi.is_unique:
        raise ValueError("a prediction file holds a duplicated row key")
    pos = fi.get_indexer(key_index(rows))
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} evaluated rows have no prediction")
    s = np.asarray(scores, dtype=float)[pos]
    if not np.all(np.isfinite(s)):
        raise ValueError("a non-finite prediction on an evaluated row")
    return s


# ------------------------------------------------------------------ the evaluation cohort
def untyped_tracks(events: pd.DataFrame) -> set:
    """The cells whose death was not typed (apoptosis or lytic)."""
    return set(events.loc[events["event_type"] == LABEL_UNTYPED, "track_id"].astype(str))


def remove_untyped(df: pd.DataFrame, tracks: set) -> tuple[pd.DataFrame, dict]:
    """Every row of the given cells removed; the counts removed per group (rows as landmarks, and cells)."""
    m = df["track_id"].astype(str).isin(tracks).to_numpy()
    rem = df[m]
    counts = {g: {"landmarks": int(len(d)), "tracks": int(d["track_id"].nunique())}
              for g, d in rem.groupby("group")} if "group" in df else {}
    return df[~m].reset_index(drop=True), {"per_group": counts, "landmarks": int(m.sum()),
                                           "tracks": int(rem["track_id"].nunique())}


def apply_cohort(rows: pd.DataFrame, untyped: set) -> tuple[pd.DataFrame, dict]:
    """The common evaluation cohort: every row of the untyped-death cells left out, with the counts removed."""
    kept, removed = remove_untyped(rows, untyped)
    if kept["track_id"].astype(str).isin(untyped).any():
        raise AssertionError("an untyped-death cell is still in the cohort")
    return kept, removed


# ------------------------------------------------------------------ the sets P, Q, N, U
def window_sets(label, cause: str) -> dict:
    """The row labels at one horizon as boolean sets: P the cause, Q the other cause, N event-free through the
    horizon, U follow-up ends inside the horizon. An untyped-death row raises (the cohort removes those cells)."""
    lab = np.asarray(label).astype(str)
    bad = sorted(set(lab) - set(WINDOW_LABELS))
    if bad:
        raise ValueError(f"unknown window label(s) {bad}")
    if (lab == LABEL_UNTYPED).any():
        raise ValueError("an UNTYPED_DEATH window is in the evaluated rows: apply the cohort first")
    return {"P": lab == cause, "Q": lab == OTHER[cause], "N": lab == LABEL_EVENT_FREE, "U": lab == LABEL_CENSORED}


def set_counts(sets: dict) -> dict:
    return {k: int(np.asarray(v).sum()) for k, v in sets.items()}


def table1_rows(windows: pd.DataFrame, untyped: set, horizon: int = TABLE1_HORIZON) -> dict:
    """Table 1's rows from the bundle's rows table: the primary panel (``primary_439``) without the movie left out
    (DU145_st_2), one panel only; the full ``windows.parquet`` holds two panels and would count rows twice.
    ``original``: rows with a determined label, an untyped-death row scored as a negative for both causes;
    ``cohort``: every row of the untyped-death cells left out, U excluded and counted. Columns KEY, group, label."""
    w = windows[windows["horizon_min"] == horizon]
    w = w[KEY + ["group", "label"]].sort_values(KEY).reset_index(drop=True)
    orig = w[w["label"] != LABEL_CENSORED].reset_index(drop=True)
    coh_all, removed = apply_cohort(w, untyped)
    coh = coh_all[coh_all["label"] != LABEL_CENSORED].reset_index(drop=True)
    u = coh_all[coh_all["label"] == LABEL_CENSORED]
    orig_lab = orig["label"].replace({LABEL_UNTYPED: LABEL_EVENT_FREE})     # scored as a negative for both causes
    return {"original": orig.assign(label=orig_lab.to_numpy()), "cohort": coh,
            "counts": {"original_rows": len(orig), "cohort_rows": len(coh),
                       "original_untyped_windows_scored_negative": int((orig["label"] == LABEL_UNTYPED).sum()),
                       "cohort_removed": removed,
                       "U_excluded_per_group": {g: int(v) for g, v in u.groupby("group").size().items()},
                       "U_excluded": len(u)}}
