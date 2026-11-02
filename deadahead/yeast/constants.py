"""Constants and folds of the budding-yeast death probe (DetecDiv, Aspert et al. 2022, Zenodo 6078462, CC BY 4.0).

One mother cell per trap; a landmark every 30 min; leave-one-position-out over 7 microscope positions.
"""

from __future__ import annotations

POSITIONS = ("Pos16_7", "Pos3_3", "Pos0_1", "Pos20_6", "Pos7_2", "Pos9_5", "Pos8_4")
TRAPS_BY_POSITION = {"Pos16_7": 50, "Pos3_3": 50, "Pos0_1": 39, "Pos20_6": 36, "Pos7_2": 34, "Pos9_5": 21,
                     "Pos8_4": 20}
DEATHS_BY_POSITION = {"Pos16_7": 40, "Pos3_3": 36, "Pos0_1": 35, "Pos20_6": 19, "Pos7_2": 18, "Pos9_5": 11,
                      "Pos8_4": 14}
N_LIFESPANS = 249                          # one of the 250 traps never shows a cell
N_LANDMARKS = 24346
HORIZONS = (30, 60, 120, 240)              # minutes
PRIMARY_H = 30
FRAME_MIN = 5                              # minutes per frame
LANDMARK_STEP = 6                          # a landmark every 6 frames (30 min)
#: 16 frames at t - round(24 i / 15), i = 0..15, oldest first (120 min of history).
CLIP_OFFSETS = (24, 22, 21, 19, 18, 16, 14, 13, 11, 10, 8, 6, 5, 3, 2, 0)
TREND_FRAMES = 12                          # the 1-h trend: value at t minus value at t - 12 frames
CROP_SHAPE = (60, 59, 3)                   # each trap video is 60 x 59 x 3 x T, uint8

SEEDS = (2601562, 2601563, 2601564)
SEED = SEEDS[0]
#: The two starting points: the released fine-tuned VideoMAE of one held-out group, and the public VideoMAE weights.
STARTS = ("videomae_finetuned", "videomae_public")
START_GROUP = "PNT1A_bp"                   # the held-out group whose fine-tuned VideoMAE is the first start
STEPS = 1000
BACKBONE_LR, LAYER_DECAY, HEAD_LR = 5e-5, 0.9, 1e-3
WEIGHT_DECAY, WARMUP_FRACTION, GRAD_CLIP = 0.05, 0.05, 1.0
CLASSES = ("NO_DEATH_30", "DEATH_30")      # no death within 30 min, death within 30 min
PRED_BATCH = 64
LABEL_EFFICIENCY_POSITIONS = 3             # the label-efficiency runs train on 3 of the 6 training positions


def fold_index(position: str) -> int:
    if position not in POSITIONS:
        raise ValueError(f"unknown position {position!r}")
    return POSITIONS.index(position)


def training_positions(heldout: str) -> tuple[str, ...]:
    fold_index(heldout)
    return tuple(p for p in POSITIONS if p != heldout)


def threshold_position(heldout: str) -> str:
    """The training position with the most deaths (Pos16_7 in 6 folds; Pos3_3 in the Pos16_7 fold)."""
    tr = training_positions(heldout)
    return max(tr, key=lambda p: (DEATHS_BY_POSITION[p], -POSITIONS.index(p)))


def threshold_run_positions(heldout: str) -> tuple[str, ...]:
    """The 5 training positions other than the threshold position."""
    thr = threshold_position(heldout)
    return tuple(p for p in training_positions(heldout) if p != thr)
