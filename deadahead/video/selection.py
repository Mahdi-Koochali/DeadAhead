"""Choose the fine-tuning setting of each fold on its selection group.

``score``: run one grid checkpoint (seed 2601562, trained without the selection group) over the
selection group's 30-minute rows and record that group's lytic AUPRC. ``choose``: rank the four
settings of a fold by that score. Non-finite scores rank last and an exact tie goes to the earlier
setting in the order (5e-5, 0.75), (5e-5, 0.9), (2e-4, 0.75), (2e-4, 0.9). The winner is retrained
with ``finetune --final`` for seeds 2601562, 2601563, 2601564.

    deadahead video-select score --fold DU145_bp --backbone-lr 2e-4 --layer-decay 0.9
    deadahead video-select choose --fold DU145_bp
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from deadahead import paths
from deadahead.evaluation.metrics import auprc

from .clips import default_clips_dir, default_store, default_windows, input_missing
from .finetune import (
    CLASSES,
    FINAL_SEEDS,
    GRID,
    cell_tag,
    checkpoint_path,
    inner_validation_group,
    label_table,
)
from .finetune import default_out_dir as default_finetune_dir
from .test_time import FOLD_ORDER

GRID_SEED = 2601562
FORWARD_BATCH = 64


def default_out_dir() -> Path:
    return paths.runs_dir() / "videomae" / "select"


def auprc_by_cause(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """One-vs-rest AUPRC of each cause's own probability; ``p`` is ``(N, 3)`` in ``CLASSES`` order."""
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(y):
        raise ValueError(f"expected p of shape (N, 3) with N = {len(y)}, got {p.shape}")
    out = {}
    for name, code in (("APOPTOSIS", 1), ("LYTIC", 2)):
        pos = (y == code)
        if not pos.any():
            raise ValueError(f"no {name} events in these rows: its AUPRC is undefined")
        if pos.all():
            raise ValueError(f"every row is {name}: its AUPRC is undefined")
        out[name] = float(auprc(pos.astype(int), p[:, code]))
    return out


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def rank_cells(scores: dict[tuple[float, float], float | None]) -> list[tuple[float, float]]:
    """Settings in order of preference: finite scores first, highest first, ties by ``GRID`` order;
    non-finite scores last, in ``GRID`` order."""
    pos = {c: i for i, c in enumerate(GRID)}

    def key(c):
        v = scores[c]
        return (0, -float(v), pos[c]) if _finite(v) else (1, 0.0, pos[c])

    return sorted(scores, key=key)


def score_path(out_dir: Path, fold: str, lr: float, ld: float) -> Path:
    return out_dir / f"score__{fold}__{cell_tag(lr, ld)}__s{GRID_SEED}.json"


def score_cell(fold: str, lr: float, ld: float, *, clips_dir: Path, store: Path, windows: Path,
               runs_dir: Path, out_dir: Path, snapshot: Path | None = None, device: str | None = None,
               batch: int = FORWARD_BATCH) -> dict:
    """Score one grid checkpoint on the fold's selection group and write its score file."""
    import pandas as pd

    from .model import image_stats, load_classifier, pretrained_snapshot, resolve_device
    from .predict import predict_probs

    idx = pd.read_parquet(Path(clips_dir) / f"clips__{fold}.parquet")
    if fold in set(idx["group"]):
        raise ValueError(f"the clip index of fold {fold} holds rows of the held-out group")
    labels = label_table(windows)
    train_groups = sorted(set(idx["group"]))
    inner = inner_validation_group(labels, train_groups)
    if inner == fold or inner not in train_groups:
        raise ValueError(f"the selection group {inner!r} is not a training group of fold {fold}")
    rows = idx[idx["group"] == inner].merge(
        labels, on=["movie_id", "track_id", "t_obs_frame", "group"], how="inner").reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"no labelled clips for selection group {inner}")

    ckpt = checkpoint_path(runs_dir, fold, lr, ld, GRID_SEED, final=False)
    snap = snapshot or pretrained_snapshot()
    mean, std = image_stats(snap)
    dev = resolve_device(device)
    model = load_classifier(ckpt, snap, dev)
    p = predict_probs(model, rows, store, mean, std, dev, batch=batch, check_finite=False)
    finite = bool(np.isfinite(p).all())
    by_cause = auprc_by_cause(rows["y"].to_numpy(), p) if finite else {"LYTIC": None, "APOPTOSIS": None}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_row = rows[["movie_id", "track_id", "t_obs_frame"]].copy()
    per_row["label"] = [CLASSES[int(v)] for v in rows["y"]]
    per_row["p_none"], per_row["p_apoptosis"], per_row["p_lytic"] = p[:, 0], p[:, 1], p[:, 2]
    per_row.to_parquet(out_dir / f"rows__{fold}__{cell_tag(lr, ld)}.parquet", index=False)
    rec = {"fold": fold, "cell": cell_tag(lr, ld), "backbone_lr": lr, "layer_decay": ld, "seed": GRID_SEED,
           "selection_group": inner, "n_rows": int(len(rows)), "n_lytic": int((rows["y"] == 2).sum()),
           "inner_lytic_auprc": by_cause["LYTIC"], "inner_apoptosis_auprc": by_cause["APOPTOSIS"],
           "all_probabilities_finite": finite}
    score_path(out_dir, fold, lr, ld).write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def choose(fold: str, out_dir: Path) -> dict | None:
    """The fold's winning setting from its score files; None while a setting is still unscored."""
    scores = {}
    for lr, ld in GRID:
        p = score_path(Path(out_dir), fold, lr, ld)
        if not p.exists():
            return None
        v = json.loads(p.read_text()).get("inner_lytic_auprc")
        scores[(lr, ld)] = v if _finite(v) else None
    order = rank_cells(scores)
    w = order[0]
    return {"fold": fold, "winner": {"backbone_lr": w[0], "layer_decay": w[1], "tag": cell_tag(*w)},
            "ranking": [cell_tag(*c) for c in order],
            "scores": {cell_tag(*c): scores[c] for c in order},
            "final_runs": [{"fold": fold, "backbone_lr": w[0], "layer_decay": w[1], "seed": s}
                           for s in FINAL_SEEDS]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Select the fine-tuning setting of a fold.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score one grid checkpoint on the selection group")
    s.add_argument("--fold", required=True, choices=FOLD_ORDER)
    s.add_argument("--backbone-lr", type=float, required=True, choices=[5e-5, 2e-4])
    s.add_argument("--layer-decay", type=float, required=True, choices=[0.75, 0.9])
    s.add_argument("--batch", type=int, default=FORWARD_BATCH)
    s.add_argument("--snapshot", type=Path, default=None)
    s.add_argument("--clips-dir", type=Path, default=None)
    s.add_argument("--store", type=Path, default=None)
    s.add_argument("--windows", type=Path, default=None)
    s.add_argument("--runs-dir", type=Path, default=None, help="fine-tuning checkpoints")
    s.add_argument("--out-dir", type=Path, default=None)
    s.add_argument("--device", default=None)
    c = sub.add_parser("choose", help="rank the four settings of a fold")
    c.add_argument("--fold", required=True, choices=FOLD_ORDER)
    c.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    out_dir = a.out_dir or default_out_dir()
    if a.cmd == "score":
        clips_dir, windows = a.clips_dir or default_clips_dir(), a.windows or default_windows()
        runs_dir = a.runs_dir or default_finetune_dir()
        ckpt = checkpoint_path(runs_dir, a.fold, a.backbone_lr, a.layer_decay, GRID_SEED, final=False)
        if (input_missing(Path(clips_dir) / f"clips__{a.fold}.parquet", "video-clips")
                or input_missing(windows, "data-windows") or input_missing(ckpt, "video-finetune")):
            return 3
        rec = score_cell(a.fold, a.backbone_lr, a.layer_decay, clips_dir=clips_dir,
                         store=a.store or default_store(), windows=windows,
                         runs_dir=runs_dir, out_dir=out_dir,
                         snapshot=a.snapshot, device=a.device, batch=a.batch)
        print(json.dumps(rec, indent=2))
        return 0
    sel = choose(a.fold, out_dir)
    if sel is None:
        print(f"fold {a.fold}: not every setting is scored yet", file=sys.stderr)
        return 3
    (Path(out_dir) / f"selected__{a.fold}.json").write_text(json.dumps(sel, indent=2) + "\n")
    print(json.dumps(sel, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
