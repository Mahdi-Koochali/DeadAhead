"""Read the predictions bundle: the forecast rows, the cell event types and the held-out predictions per method.

Layout (see MANIFEST.json in the bundle):
    rows.parquet                  movie_id, track_id, t_obs_frame, group, label, horizon_min, window_weight
    events.parquet                track_id, event_type, t_event (death frame); event_type is LYTIC, APOPTOSIS,
                                  UNTYPED_DEATH (a death without a type), CENSORED (no death observed) or EXCLUDED
                                  (a typed death dated after the movie ends)
    movies.parquet                movie_id, num_frames (warnings only)
    predictions/<method>.parquet  group, seed, cause, movie_id, track_id, t_obs_frame, score
    inner_predictions.parquet     outer_group, seed, inner_group, cause, movie_id, track_id, t_obs_frame, score
                                  (DeadAhead's cross-fitted inner heads, deadahead train-head --inner-group;
                                  warnings only)
    inner_predictions/<method>.parquet
                                  the same columns for another method's cross-fitted inner heads (the image
                                  baseline's warnings)
    landmarks.parquet             movie_id, track_id, t_obs_frame, group: the forecast rows of the survival
                                  population (gap analysis)
    cumulative_incidence.parquet  method, group, seed, movie_id, track_id, t_obs_frame, F_<cause>_<h> for h = 60
                                  and 120: the stored cumulative incidences of the four survival methods (also the
                                  window analysis)
    cumulative_incidence_30_240.parquet
                                  the same rows for h = 30 and 240 (fixed-horizon results, window analysis)
    inner_cumulative_incidence.parquet
                                  outer_group, seed, inner_group, movie_id, track_id, t_obs_frame, F_<cause>_<h> for
                                  h = 60, 120 and 240: DeadAhead's cross-fitted inner heads, deadahead train-head
                                  --inner-group (fixed-horizon results)
    decomposition/videomae_adapted_survival.parquet
                                  group, seed, cause, movie_id, track_id, t_obs_frame, score: the survival head on
                                  the adapted, not fine-tuned, VideoMAE features and the handcrafted features
    decomposition/test_time.parquet
                                  condition, group, seed, cause, movie_id, track_id, t_obs_frame, score: the
                                  fine-tuned video model with the tracked cell hidden, its 16 frames shuffled or its
                                  last frame repeated 16 times
    decomposition/oracle_labels.parquet
                                  group, movie_id, track_id, t_obs_frame, label: the held-out landmarks whose 30-min
                                  outcome is known in the complete movie, with the oracle label (0 no event, 1
                                  apoptosis, 2 lytic death)
    censoring/predictions.parquet group, seed, regime, horizon_min, label, score_APOPTOSIS, score_LYTIC: the
                                  held-out oracle labels and scores of the injected-censoring experiment
    censoring/units.parquet       group, seed, regime, achieved_censoring_rate and the survival regime's risk-set
                                  counts, one row per unit
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from deadahead.evaluation.rows import untyped_tracks


#: bundle file -> key in the loaded bundle (the supplementary parts that read them)
OPTIONAL_FILES = {"decomposition/videomae_adapted_survival.parquet": "adapted_predictions",
                  "decomposition/test_time.parquet": "test_time_predictions",
                  "decomposition/oracle_labels.parquet": "oracle_labels",
                  "censoring/predictions.parquet": "censoring_predictions",
                  "censoring/units.parquet": "censoring_units"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_manifest(bundle: Path) -> dict:
    p = Path(bundle) / "MANIFEST.json"
    if not p.is_file():
        raise FileNotFoundError(f"no MANIFEST.json in {bundle}")
    return json.loads(p.read_text())


def check_files(bundle: Path, manifest: dict) -> list[str]:
    """Files that are missing or whose size (MANIFEST "bytes") or sha256 differs from the manifest."""
    out = []
    for rel, info in manifest["files"].items():
        p = Path(bundle) / rel
        if not p.is_file():
            out.append(f"{rel}: missing")
        elif "bytes" in info and p.stat().st_size != int(info["bytes"]):
            out.append(f"{rel}: size differs from the manifest")
        elif "sha256" in info and sha256_file(p) != info["sha256"]:
            out.append(f"{rel}: sha256 differs from the manifest")
    return out


def load_bundle(bundle: Path, verify: bool = True) -> dict:
    """{"rows", "events", "untyped", "predictions": {method: frame}, "manifest"} from a bundle folder."""
    bundle = Path(bundle)
    manifest = read_manifest(bundle)
    if verify:
        bad = check_files(bundle, manifest)
        if bad:
            raise ValueError("the bundle does not match its manifest: " + "; ".join(bad[:5]))
    rows = pd.read_parquet(bundle / "rows.parquet")
    events = pd.read_parquet(bundle / "events.parquet")
    preds = {}
    for rel, info in manifest["files"].items():
        if rel.startswith("predictions/"):
            preds[info["method"]] = pd.read_parquet(bundle / rel)
    out = {"rows": rows, "events": events, "untyped": untyped_tracks(events), "predictions": preds,
           "manifest": manifest}
    for name in ("movies", "inner_predictions", "landmarks", "cumulative_incidence", "cumulative_incidence_30_240",
                 "inner_cumulative_incidence"):
        rel = f"{name}.parquet"
        if rel in manifest["files"]:
            out[name] = pd.read_parquet(bundle / rel)
    for rel, name in OPTIONAL_FILES.items():
        if rel in manifest["files"]:
            out[name] = pd.read_parquet(bundle / rel)
    inner = {info["method"]: pd.read_parquet(bundle / rel) for rel, info in manifest["files"].items()
             if rel.startswith("inner_predictions/")}
    if inner:
        out["inner_predictions_by_method"] = inner
    return out
