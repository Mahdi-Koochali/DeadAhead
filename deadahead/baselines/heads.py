"""LightGBM heads of the comparison methods of Table 1, one held-out group at a time.

    handcrafted                    the 112 handcrafted history features
    dinov2                         DINOv2 embeddings of 9 single frames over the past 120 min, pooled as last, mean
                                   and change (3 x 768)
    dinov2_handcrafted             both blocks
    dinov2_finetuned_handcrafted   the label-fine-tuned DINOv2 embeddings and the handcrafted features

Per cause, a binary LightGBM classifier (300 rounds, fixed settings, no early stopping) is fitted on the 30-min rows
of the eight training groups, y = label == cause, weight window_weight, and scores every 30-min row of the held-out
group, once per seed. Rows censored inside the horizon are not forecast rows. The ``refit`` policy (the Table 1 entry)
leaves the training rows with an untyped death inside the horizon out of training; ``original`` keeps them as
negatives. Each image method reads the embedding store of the encoder adapted without the held-out group.
``--inner`` adds the cross-fitted scores inside the eight training groups (one head per training group h, fitted on
the other seven), from which the warnings of an image method take training-only alarm thresholds.

    deadahead baseline-heads --method dinov2_handcrafted --group DU145_bp
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from deadahead.evaluation.metrics import auprc, auroc
from deadahead.features.history import (
    HISTORY_FRAMES_W,
    history_columns,
    history_features,
)
from deadahead.paths import runs_dir, tables_dir
from deadahead.survival.frames import EXCLUDED_MOVIE, GROUPS, PANEL_NAME, SEEDS
from deadahead.survival.head import usable_cpus

HORIZON = 30
CAUSES = ("APOPTOSIS", "LYTIC")
UNTYPED = "UNTYPED_DEATH"            # the window label of an untyped death inside (t, t + h]
KEYS = ["movie_id", "track_id", "t_obs_frame"]
ROW_COLS = ["example_id", "movie_id", "track_id", "t_obs_frame", "group", "horizon_min", "label", "window_weight"]
POLICIES = ("refit", "original")
TEMPORAL = {"n_timesteps": 9, "stride_frames": 5}    # 9 frames, 5 frames (15 min) apart: t - 40, ..., t
EXCLUDED_MOVIES = (EXCLUDED_MOVIE,)

#: method -> (embedding store kind or None, input layer)
METHODS = {
    "handcrafted": (None, "handcrafted"),
    "dinov2": ("dinov2", "dino"),
    "dinov2_handcrafted": ("dinov2", "both"),
    "dinov2_finetuned_handcrafted": ("dinov2_finetuned", "both"),
}


# Fixed settings, the same for every method, held-out group and seed. The released heads were fitted with 16 threads;
# the scores were measured to be the same at other thread counts to about 1e-17.
LGB = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
       "feature_fraction": 0.3, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
       "num_threads": int(os.environ.get("OMP_NUM_THREADS") or usable_cpus())}
ROUNDS = 300


def set_threads(n: int) -> None:
    """Set the number of LightGBM threads of the heads."""
    LGB["num_threads"] = int(n)


def default_store(kind: str, group: str) -> Path:
    """The per-frame embedding store of the encoder that held out ``group``: <movie>.npy and <movie>.parquet."""
    return runs_dir() / kind / "embeddings" / group


def default_out_dir() -> Path:
    return runs_dir() / "baselines" / "heads"


def unit_name(method: str, group: str, policy: str) -> str:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r} ({POLICIES})")
    return f"{method}__{group}__{policy}"


# ------------------------------------------------------------------ the table
def forecast_rows(windows: pd.DataFrame, excluded=EXCLUDED_MOVIES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows, landmarks): the primary-panel rows with a determined label, without the excluded movie, and one
    landmark (movie, track, frame, cell number) per forecast time."""
    w = windows[(windows["panel"] == PANEL_NAME) & (windows["label"] != "CENSORED_IN_HORIZON")]
    w = w[~w["movie_id"].isin(list(excluded))]
    w["cell_num"] = w["track_id"].str.split(":").str[-1].astype(int)
    lm = w.drop_duplicates(subset=["movie_id", "track_id", "t_obs_frame"])[
        ["movie_id", "track_id", "t_obs_frame", "cell_num"]].reset_index(drop=True)
    return w, lm


def pool_embeddings(landmarks: pd.DataFrame, temporal: dict, store: Path) -> tuple[np.ndarray, list[str]]:
    """Per landmark, the embeddings of its 9 frames pooled into last, mean and change (last - first): (n, 3 D).

    ``store`` holds <movie>.npy (one row per crop) and <movie>.parquet (cell_num, frame_index, row). A frame without a
    crop contributes a zero vector.
    """
    T = int(temporal["n_timesteps"])
    stride = int(temporal["stride_frames"])

    store_, index = {}, {}
    for m in sorted(landmarks["movie_id"].unique()):
        store_[m] = np.load(store / f"{m}.npy", mmap_mode="r")
        idx = pd.read_parquet(store / f"{m}.parquet")
        index[m] = {(int(c), int(f)): int(r)
                    for c, f, r in zip(idx["cell_num"], idx["frame_index"], idx["row"])}

    D = store_[next(iter(store_))].shape[1]
    out = np.zeros((len(landmarks), 3 * D), dtype=np.float32)
    for i, rec in enumerate(landmarks.itertuples(index=False)):
        arr, ix = store_[rec.movie_id], index[rec.movie_id]
        frames = [rec.t_obs_frame - stride * (T - 1 - k) for k in range(T)]
        rows = [ix.get((rec.cell_num, f)) for f in frames]
        seq = np.stack([arr[r] if r is not None else np.zeros(D, dtype=np.float16) for r in rows])
        seq = seq.astype(np.float32)
        out[i, :D] = seq[-1]
        out[i, D:2 * D] = seq.mean(axis=0)
        out[i, 2 * D:] = seq[-1] - seq[0]
    cols = ([f"dino_last_{j}" for j in range(D)] + [f"dino_mean_{j}" for j in range(D)]
            + [f"dino_delta_{j}" for j in range(D)])
    return out, cols


def build_table(w: pd.DataFrame, lm: pd.DataFrame, hist: pd.DataFrame, store: Path | None,
                temporal: dict = TEMPORAL) -> tuple[pd.DataFrame, list[str]]:
    """The rows joined to the pooled embeddings of ``store`` (none when ``store`` is None) and to the history
    features. Returns (table, embedding columns)."""
    if store is None:
        data = w.merge(hist, left_on=["track_id", "t_obs_frame"], right_on=["track_id", "frame_index"], how="inner")
        return data, []
    emb, dcols = pool_embeddings(lm, temporal, Path(store))
    lm_emb = pd.concat([lm, pd.DataFrame(emb, columns=dcols)], axis=1)
    data = (w.merge(lm_emb.drop(columns=["cell_num"]), on=["movie_id", "track_id", "t_obs_frame"])
            .merge(hist, left_on=["track_id", "t_obs_frame"],
                   right_on=["track_id", "frame_index"], how="inner"))
    return data, dcols


def input_columns(layer: str, dcols: list, pcols: list) -> list:
    """Embedding columns, handcrafted columns, or both (embeddings first)."""
    return {"dino": dcols, "both": dcols + pcols, "handcrafted": pcols}[layer]


# ------------------------------------------------------------------ one held-out group
def refit_fold(data, cols: list[str], g: str, seeds: list[int], causes=CAUSES, *, drop_untyped: bool = False):
    """Fit the head on every group but ``g`` and score g's 30-min rows, both causes, once per seed.

    Returns (rows: one per held-out row, cause and seed with its score; per-cause AUPRC and AUROC per seed and their
    means). With ``drop_untyped`` the training rows labelled UNTYPED_DEATH are left out of training; the held-out
    rows are unchanged.
    """
    dh = data[data["horizon_min"] == HORIZON]
    te, tr = dh[dh["group"] == g], dh[dh["group"] != g]
    if te.empty or tr.empty:
        raise ValueError(f"fold {g}: {len(te)} held-out and {len(tr)} training rows at h{HORIZON}")
    removed = tr[tr["label"] == UNTYPED] if drop_untyped else tr.iloc[0:0]
    if drop_untyped:
        tr = tr[tr["label"] != UNTYPED]
    parts, res = [], {}
    for cause in causes:
        per_seed = []
        y = (tr["label"] == cause).to_numpy(dtype=int)
        if y.sum() == 0:
            raise ValueError(f"fold {g}: no {cause} training row")
        yt = (te["label"] == cause).to_numpy(dtype=float)
        for seed in seeds:
            m = lgb.train({**LGB, "seed": seed}, lgb.Dataset(tr[cols], label=y, weight=tr["window_weight"]),
                          num_boost_round=ROUNDS)
            p = m.predict(te[cols])
            per_seed.append({"seed": int(seed), "auprc": float(auprc(yt, p)), "auroc": float(auroc(yt, p))})
            part = te[[c for c in ROW_COLS if c in te.columns]].copy()
            part["cause"], part["seed"], part["score"] = cause, int(seed), p
            parts.append(part)
            print(f"    {g:<9} {cause:<10} seed {seed}  rows {len(te):>5}  AUPRC {per_seed[-1]['auprc']:.10f}",
                  flush=True)
        res[cause] = {"per_seed": per_seed,
                      "auprc_mean": float(np.mean([s["auprc"] for s in per_seed])),
                      "auroc_mean": float(np.mean([s["auroc"] for s in per_seed])),
                      "n_events": int(te.loc[te["label"] == cause, "track_id"].nunique()),
                      "n_rows": int(len(te)), "n_training_rows": int(len(tr)),
                      "removed_training_windows": {
                          "n": int(len(removed)), "weight": float(removed["window_weight"].sum()),
                          "tracks": int(removed["track_id"].nunique()),
                          "by_group": {k: int(v) for k, v in removed.groupby("group").size().items()},
                          "example_ids": sorted(str(x) for x in removed.get("example_id", pd.Series(dtype=str))),
                          "label_for_this_cause_without_removal": "negative (y = 0)"}}
    return pd.concat(parts, ignore_index=True), res


def inner_fits(data, cols: list[str], g: str, seeds, causes=CAUSES, inner_groups=None) -> pd.DataFrame:
    """Cross-fitted scores inside the training groups of ``g`` (refit policy): for each training group h (all eight,
    or ``inner_groups``), the head fitted on the seven groups other than g and h scores h's 30-min rows. Used for
    training-only alarm thresholds."""
    inner = data[data["group"] != g]
    parts = []
    for h in inner_groups or [x for x in GROUPS if x != g]:
        if h == g:
            raise ValueError(f"the inner group {h} is the held-out group")
        rows, _ = refit_fold(inner, cols, h, list(seeds), causes, drop_untyped=True)
        parts.append(rows.assign(outer_group=g))
    return pd.concat(parts, ignore_index=True)


def run(method: str, group: str, policies, seeds, *, store: Path | None, windows: pd.DataFrame,
        hist: pd.DataFrame, out_dir: Path, causes=CAUSES, inner: bool = False) -> dict:
    """Build the table once and fit every requested policy; write <unit>.parquet (per-row scores) and <unit>.json
    (per-cause AUPRC and AUROC on the held-out group, the removed training rows, the settings). With ``inner`` also
    <unit>__inner.parquet, the cross-fitted scores with the bundle's ``outer_group`` and ``inner_group`` columns."""
    kind, layer = METHODS[method]
    if kind is not None and store is None:
        store = default_store(kind, group)
    if kind is not None and not Path(store).is_dir():
        raise FileNotFoundError(f"embedding store {store}")
    t0 = time.time()
    w, lm = forecast_rows(windows)
    data, dcols = build_table(w, lm, hist, store if kind is not None else None)
    cols = input_columns(layer, dcols, history_columns(hist))
    t_table = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for policy in policies:
        t1 = time.time()
        rows, res = refit_fold(data, cols, group, list(seeds), causes, drop_untyped=(policy == "refit"))
        unit = unit_name(method, group, policy)
        pq = out_dir / f"{unit}.parquet"
        tmp = pq.with_name(pq.name + ".tmp")
        rows.to_parquet(tmp, index=False)
        os.replace(tmp, pq)
        rec = {"method": method, "group": group, "policy": policy, "horizon_min": HORIZON, "seeds": [int(s) for s in seeds],
               "store": None if kind is None else str(store), "n_features": len(cols),
               "lightgbm_params": dict(LGB), "rounds": ROUNDS, "results": res,
               "seconds": {"table": round(t_table, 1), "fit": round(time.time() - t1, 1)},
               "rows_file": pq.name, "n_rows_written": int(len(rows))}
        (out_dir / f"{unit}.json").write_text(json.dumps(rec, indent=2, default=float) + "\n")
        summary[unit] = {c: res[c]["auprc_mean"] for c in res}
        print(f"{unit}: AUPRC " + ", ".join(f"{c} {v:.4f}" for c, v in summary[unit].items()) + f" -> {pq}", flush=True)
    if inner:
        rows = inner_fits(data, cols, group, list(seeds), causes).rename(columns={"group": "inner_group"})
        pq = out_dir / f"{unit_name(method, group, 'refit')}__inner.parquet"
        tmp = pq.with_name(pq.name + ".tmp")
        rows.to_parquet(tmp, index=False)
        os.replace(tmp, pq)
        print(f"{pq.name}: {len(rows)} cross-fitted rows in the 8 training groups", flush=True)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit a comparison method's LightGBM head for one held-out group.")
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--group", "--fold", dest="group", required=True, choices=list(GROUPS))
    ap.add_argument("--policy", choices=[*POLICIES, "both"], default="refit",
                    help="refit (the Table 1 entry): untyped-death training rows left out; original: kept as "
                         "negatives; both (default refit)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--store", type=Path, help="embedding store of the held-out group "
                                               "(default runs/<dinov2|dinov2_finetuned>/embeddings/<group>)")
    ap.add_argument("--windows", type=Path, help="windows table (default tables/windows.parquet)")
    ap.add_argument("--cells", type=Path, help="cells table for the history features (default tables/cells.parquet)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default runs/baselines/heads")
    ap.add_argument("--threads", type=int, default=None, help="LightGBM threads (default OMP_NUM_THREADS or all)")
    ap.add_argument("--inner", action="store_true", help="also the cross-fitted refit scores inside the training "
                                                         "groups (the training-only thresholds of the warnings)")
    a = ap.parse_args(argv)
    if a.threads:
        set_threads(a.threads)
    policies = POLICIES if a.policy == "both" else (a.policy,)
    windows = pd.read_parquet(a.windows or tables_dir() / "windows.parquet")
    hist = history_features(HISTORY_FRAMES_W, a.cells)
    run(a.method, a.group, policies, a.seeds, store=a.store, windows=windows, hist=hist,
        out_dir=a.out_dir or default_out_dir(), inner=a.inner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
