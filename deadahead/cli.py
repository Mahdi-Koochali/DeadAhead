"""Command-line entry point: ``deadahead <command> [options]``."""

from __future__ import annotations

import importlib
import sys

from deadahead import __version__

COMMANDS = {
    "data-inventory": ("deadahead.data.inventory", "list the movies of the extracted archives (tables/movies.csv)"),
    "data-tables": ("deadahead.data.tables", "movies, tracks and events tables from the annotation spreadsheets"),
    "data-cells": ("deadahead.features.frame_measures", "per-cell, per-frame shape and phase measures (cells.parquet)"),
    "data-windows": ("deadahead.data.windows", "forecast rows (windows.parquet) and the leave-one-group-out folds"),
    "video-crops": ("deadahead.video.crops", "cut 128-pixel crops around every tracked cell, every frame"),
    "video-clips": ("deadahead.video.clips", "index the 16-frame clips of every forecast row, per fold"),
    "video-adapt": ("deadahead.video.adapt", "masked-autoencoder adaptation of VideoMAE for one held-out group"),
    "video-finetune": ("deadahead.video.finetune", "fine-tune VideoMAE for one held-out group, setting and seed"),
    "video-select": ("deadahead.video.selection", "score the four settings on the inner group and choose one"),
    "video-predict": ("deadahead.video.predict", "held-out class probabilities of a fine-tuned VideoMAE"),
    "video-features": ("deadahead.video.features", "pooled 1,024-d VideoMAE features for every forecast row"),
    "baseline-dinov2-adapt": ("deadahead.baselines.dinov2_adapt", "adapt DINOv2 without labels on image tiles of the training groups"),
    "baseline-dinov2-embed": ("deadahead.baselines.dinov2_embed", "per-frame DINOv2 embeddings of every forecast row"),
    "baseline-dinov2-finetune": ("deadahead.baselines.dinov2_finetune", "fine-tune DINOv2's last two blocks on the death labels"),
    "baseline-heads": ("deadahead.baselines.heads", "LightGBM heads of the image baselines and Handcrafted"),
    "baseline-dinov2-survival": ("deadahead.baselines.dinov2_survival", "DINOv2 + handcrafted through DeadAhead's survival head"),
    "baseline-deephit": ("deadahead.baselines.deephit", "DeepHit on DeadAhead's inputs, setting chosen on the inner group"),
    "video-test-time": ("deadahead.video.test_time", "test-time changes: hide the cell, shuffle the frames, repeat the last frame"),
    "censoring-experiment": ("deadahead.survival.censoring", "the censoring experiment: train one unit with injected censoring"),
    "history-features": ("deadahead.features.history", "the 112 handcrafted features over the past 120 min"),
    "train-head": ("deadahead.survival.train", "train (or apply) the competing-risks survival head for one fold and seed"),
    "yeast-data": ("deadahead.yeast.data", "yeast probe: lifespan labels, rows and pixel store from the DetecDiv archive"),
    "yeast-finetune": ("deadahead.yeast.finetune", "yeast probe: fine-tune VideoMAE on yeast death"),
    "yeast-predict": ("deadahead.yeast.predict", "yeast probe: predictions of a saved model, held-out or zero-shot"),
    "yeast-survival": ("deadahead.yeast.survival", "yeast probe: the survival layer on the embeddings"),
    "yeast-rules": ("deadahead.yeast.rules", "yeast probe: the simple rules (labels, brightness, size) on each held-out position"),
    "yeast-score": ("deadahead.yeast.score", "yeast probe: recompute every score from the yeast bundle"),
    "reproduce": ("deadahead.reproduce.main", "recompute the paper's results from the predictions bundle"),
    "plot-figures": ("deadahead.report.figures:main_all", "draw every figure of the supplement from the reproduce output"),
    "plot-warnings": ("deadahead.report.figures", "draw Fig. 2 from the warnings table with its spread"),
    "plot-gap-window": ("deadahead.report.figures:main_gap_window", "draw the post hoc window figure of S8"),
    "plot-alarm-rate": ("deadahead.report.figures:main_alarm_rate", "draw the post hoc alarm-rate figure of S8"),
}


def _usage() -> str:
    width = max(map(len, COMMANDS))
    lines = ["usage: deadahead <command> [options]", "       deadahead --version", "", "commands:"]
    lines += [f"  {name:<{width}}  {help_}" for name, (_, help_) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    if argv[0] == "--version":
        print(f"deadahead {__version__}")
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"unknown command: {name}\n\n{_usage()}", file=sys.stderr)
        return 2
    module_name, _, entry = COMMANDS[name][0].partition(":")
    module = importlib.import_module(module_name)
    saved = list(sys.argv)
    sys.argv[:1] = [f"deadahead {name}"]          # argparse names the program after argv[0] in usage and errors
    try:
        result = getattr(module, entry or "main")(rest)
    finally:
        sys.argv[:] = saved
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
