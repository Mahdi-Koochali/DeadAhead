"""A synthetic two-movie dataset in the CellDeathDetect file layout, for the data-preparation tests.

Each 20-frame, 64 x 64 movie holds:
    cell 1  present in every frame, alive at the end      -> censored at the end of the movie
    cell 2  shrinks and is gone from frame 12 on          -> apoptosis, onset frame 12 (1-based)
    cell 3  drifts towards the right edge                 -> censored
    cell 4  splits into labels 4 and 5 at frame 8         -> lytic death, onset frame 18 (1-based)
    cell 6  is gone from frame 10 on                      -> censored
    cell 7  is in the label spreadsheet but in no mask
Phase is 3.0 inside every cell plus a background of 0.05, 0.10 or 0.15 that cycles with the frame. No random numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

H = W = 64
N_FRAMES = 20
LABELS = [
    {"cell_num": 1, "death_type": 3, "death_frame": None},
    {"cell_num": 2, "death_type": 1, "death_frame": 12},
    {"cell_num": 3, "death_type": 3, "death_frame": None},
    {"cell_num": 4, "death_type": 2, "death_frame": 18},
    {"cell_num": 6, "death_type": 3, "death_frame": None},
    {"cell_num": 7, "death_type": 3, "death_frame": None},
]


def disc(mask: np.ndarray, cy: float, cx: float, r: float, label: int) -> None:
    """Paint a filled disc of radius ``r`` centred at (cy, cx) with ``label``."""
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    sel = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    mask[sel] = label


def stacks() -> tuple[np.ndarray, np.ndarray]:
    """(phase, mask) stacks of one synthetic movie, each (N_FRAMES, H, W)."""
    masks = np.zeros((N_FRAMES, H, W), np.uint16)
    images = np.zeros((N_FRAMES, H, W), np.float32)
    for t in range(N_FRAMES):
        m = masks[t]
        disc(m, 12, 12 + 0.1 * t, 5, 1)
        if t < 12:
            disc(m, 40, 14, 5 - 0.15 * t, 2)
        disc(m, 20, 30 + 1.8 * t, 4, 3)
        if t < 8:
            disc(m, 50, 45, 6, 4)
        else:
            disc(m, 47, 42, 4, 4)
            disc(m, 54, 49, 4, 5)
        if t < 10:
            disc(m, 30, 8, 4, 6)
        img = (m > 0).astype(np.float32) * 3.0
        img += 0.05 * ((t % 3) + 1)
        images[t] = img
    return images, masks


def write_micro_movies(raw: Path, folder: str = "QPI_annotated_timelapse_MICRO") -> list[str]:
    """Write movies MICRO_xx_1 and MICRO_xx_2 (phase, mask and label files) under ``raw/folder``; return their ids."""
    out = Path(raw) / folder
    out.mkdir(parents=True, exist_ok=True)
    images, masks = stacks()
    ids = []
    for seq in (1, 2):
        tag = f"MICRO_xx_{seq}"
        tifffile.imwrite(out / f"QPI_{tag}.tif", images)
        tifffile.imwrite(out / f"mask_{tag}.tif", masks)
        pd.DataFrame(LABELS).to_excel(out / f"labels_{tag}.xlsx", index=False)
        ids.append(tag)
    return ids
