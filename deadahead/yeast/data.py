"""Yeast data: landmark rows, 16-frame clips, the pixel store, the per-fold channel and intensity map, the picture
measurements and the per-fold row files. CPU only.

Two readers of the archive are kept apart: ``load_label_series`` reads the human classes (`deep`) and returns no
pixel; ``build_vid_store`` reads the trap videos (`vid`) only and returns no label. The archive is read in memory,
never extracted.

    deadahead yeast-data --zip <lifespanAnnotation.zip>          rows, row files, pixel store and per-fold inputs
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from deadahead.paths import data_root

from . import constants as C
from . import lifespans as L

KEY = ["unit", "t"]                       # a landmark: the trap (one mother cell per trap) and the frame t
PIC_BRIGHT = ("pic_mean", "pic_std", "pic_mean_trend", "pic_std_trend")
PIC_SIZE = PIC_BRIGHT + ("pic_size", "pic_size_trend")
CHANNEL_SAMPLE = 1000                     # frames drawn per fold for the channel choice
SOURCE_SAMPLE = 10000                     # quantitative-phase crops drawn for the source moments
SOURCE_EXCLUDED_MOVIES = ("DU145_st_2",)  # the movie left out of the quantitative-phase training
TABLE_FILES = ("classes.parquet", "brightness.parquet", "size.parquet", "folds.json")


class StoreError(RuntimeError):
    pass


def yeast_dir() -> Path:
    return data_root() / "yeast"


def trap_position(trap: str) -> str:
    return trap.rsplit("_", 1)[0]


def trap_of_member(name: str) -> str:
    return name.rsplit("/", 1)[-1].removeprefix("lstm_labeled_").removesuffix(".mat")


def timeseries_members(z: zipfile.ZipFile) -> list[str]:
    return sorted(n for n in z.namelist() if n.endswith(".mat") and "/timeseries/" in n)


# ------------------------------------------------------------------ labels
def load_label_series(zpath: Path) -> dict:
    """{trap: per-frame human class array} from the archive's `deep` series."""
    return L.load_detecdiv(Path(zpath))


def lifespan_table(series: dict) -> pd.DataFrame:
    """One row per trap with a cell: start, end (the first dead, clogged or empty frame, or the video length), event
    and the number of frames."""
    rows = []
    for trap, cls in sorted(series.items()):
        ls = L.detecdiv_lifespan(cls)
        if ls is None:
            continue
        start, end, event = ls
        rows.append({"unit": trap, "position": trap_position(trap), "start": start, "end": end, "event": bool(event),
                     "n_frames": len(cls)})
    return pd.DataFrame(rows)


def build_rows(series: dict) -> tuple[pd.DataFrame, dict]:
    """The landmark rows with the survival columns: ``ev_off`` = event frame - t (NaN when censored), ``obs_off`` =
    last observed frame - t, and ``nb_off`` = next bud frame - t (a target, never an input; NaN when none follows)."""
    rows, notes = L.detecdiv_rows(series)
    lt = lifespan_table(series).set_index("unit")
    rows = rows.rename(columns={"fold": "position"})
    end = rows["unit"].map(lt["end"]).to_numpy()
    ev = rows["unit"].map(lt["event"]).to_numpy().astype(bool)
    rows["ev_off"] = np.where(ev, end - rows["t"].to_numpy(), np.nan)
    rows["obs_off"] = (end - 1) - rows["t"].to_numpy()
    # the next bud after t inside the lifespan; NaN when no bud follows before the lifespan's end
    nb = np.full(len(rows), np.nan)
    for trap, idx in rows.groupby("unit").indices.items():
        st, en = int(lt.loc[trap, "start"]), int(lt.loc[trap, "end"])
        b = L.detecdiv_buds(series[trap], st, en)
        t = rows["t"].to_numpy()[idx]
        k = np.searchsorted(b, t, side="right")
        has = k < len(b)
        nb[idx[has]] = b[k[has]] - t[has]
    rows["nb_off"] = nb
    return rows, notes


def bud_death_bins(rows: pd.DataFrame, bin_frames: int = C.LANDMARK_STEP):
    """(event bin, cause, censor bin) with two causes, the next bud (1) and death (2), whichever comes first; a
    landmark with neither is censored as in ``survival_bins``."""
    ev_bin, cens_bin = survival_bins(rows, bin_frames)
    nb = rows["nb_off"].to_numpy(float)
    nb_bin = np.where(np.isfinite(nb) & (nb > 0), np.ceil(nb / bin_frames), np.inf)
    bud_first = nb_bin < ev_bin                        # a bud in an earlier bin than the death (or no death)
    out_bin = np.where(bud_first, nb_bin, ev_bin)
    cause = np.where(bud_first, 1, np.where(np.isfinite(ev_bin), 2, 0))
    cens = np.where(np.isfinite(out_bin), np.inf, cens_bin)
    return out_bin, cause, cens


def survival_bins(rows: pd.DataFrame, bin_frames: int = C.LANDMARK_STEP) -> tuple[np.ndarray, np.ndarray]:
    """(event bin, censor bin) per landmark on 30-min bins: bin k covers (t + 6(k-1), t + 6k]; an event at e falls in
    bin ceil((e - t) / 6); a censored landmark is at risk in bin k iff k <= floor(obs_off / 6). inf = never."""
    ev = rows["ev_off"].to_numpy(float)
    ev_bin = np.where(np.isfinite(ev) & (ev > 0), np.ceil(ev / bin_frames), np.inf)
    cens_bin = np.where(np.isfinite(ev), np.inf, np.floor(rows["obs_off"].to_numpy(float) / bin_frames))
    return ev_bin, cens_bin


# ------------------------------------------------------------------ clips
def clip_frames(t: int) -> np.ndarray:
    """The 16 frames of the clip at landmark t, oldest first; before the video's first frame the first frame is
    held. Every frame is at or before t, and the last is t."""
    f = np.maximum(int(t) - np.asarray(C.CLIP_OFFSETS, dtype=np.int64), 0)
    if f.max() != t or (np.diff(f) < 0).any():
        raise StoreError(f"clip at t={t} breaks the past-only rule: {f.tolist()}")
    return f


def clip_index(keys: pd.DataFrame) -> pd.DataFrame:
    """unit, t, f0..f15 for every landmark key (no label)."""
    fr = np.stack([clip_frames(t) for t in keys["t"].to_numpy()]) if len(keys) else np.zeros((0, 16), np.int64)
    out = keys[KEY].reset_index(drop=True).copy()
    for i in range(fr.shape[1]):
        out[f"f{i}"] = fr[:, i]
    if len(out) and not (fr.max(axis=1) == out["t"].to_numpy()).all():
        raise StoreError("a clip reads past its landmark")
    return out


# ------------------------------------------------------------------ the pixel store
def load_vid_only(raw: bytes) -> np.ndarray:
    """One .mat member's `vid` as (T, 60, 59, 3) uint8; raises unless the loaded dict holds exactly `vid`."""
    from scipy.io import loadmat
    m = loadmat(io.BytesIO(raw), variable_names=["vid"])
    keys = {k for k in m if not k.startswith("__")}
    if keys != {"vid"}:
        raise StoreError(f"the loaded keys are {sorted(keys)}, not exactly ['vid']")
    vid = np.asarray(m["vid"])
    if vid.dtype != np.uint8 or vid.ndim != 4 or tuple(vid.shape[:3]) != C.CROP_SHAPE:
        raise StoreError(f"vid has dtype {vid.dtype} and shape {vid.shape}, not uint8 60 x 59 x 3 x T")
    return np.ascontiguousarray(np.moveaxis(vid, 3, 0))            # (T, 60, 59, 3)


def build_vid_store(zpath: Path, out_dir: Path, *, traps: list[str] | None = None) -> dict:
    """Write one (T, 60, 59, 3) uint8 array per trap to ``out_dir/<trap>.npy`` (atomic; an existing file is kept, so
    an interrupted build resumes) and a manifest of the members opened."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opened, n_frames = [], {}
    with zipfile.ZipFile(zpath) as z:
        for n in timeseries_members(z):
            trap = trap_of_member(n)
            if traps is not None and trap not in traps:
                continue
            dst = out_dir / f"{trap}.npy"
            if dst.is_file():
                n_frames[trap] = int(np.load(dst, mmap_mode="r").shape[0])
                continue
            opened.append(n)
            arr = load_vid_only(z.read(n))
            tmp = dst.with_name(dst.name + f".tmp{os.getpid()}.npy")
            np.save(tmp, arr)
            os.replace(tmp, dst)
            n_frames[trap] = int(arr.shape[0])
    man = {"what": "yeast pixel store: vid only, loaded with variable_names=['vid']; no label",
           "members_opened_this_build": opened, "n_frames": n_frames, "loaded_keys": ["vid"]}
    mp = out_dir / "manifest.json"
    if mp.is_file():
        old = json.loads(mp.read_text())
        man["members_opened_earlier"] = sorted(set(old.get("members_opened_this_build", []))
                                               | set(old.get("members_opened_earlier", [])))
    mp.write_text(json.dumps(man, indent=1) + "\n")
    return man


class VidStore:
    """Clips read out of the pixel store through memory maps (numpy only)."""

    def __init__(self, store_dir: Path):
        self.dir = Path(store_dir)
        self._maps: dict[str, np.ndarray] = {}

    def video(self, trap: str) -> np.ndarray:
        m = self._maps.get(trap)
        if m is None:
            p = self.dir / f"{trap}.npy"
            if not p.is_file():
                raise StoreError(f"the store has no pixels for trap {trap} ({p})")
            m = np.load(p, mmap_mode="r")
            self._maps[trap] = m
        return m

    def clip(self, trap: str, frames, channel: int) -> np.ndarray:
        v = self.video(trap)
        f = np.asarray(frames, dtype=np.int64)
        if f.min() < 0 or f.max() >= len(v):
            raise StoreError(f"clip frames {f.min()}..{f.max()} fall outside trap {trap} ({len(v)} frames)")
        return np.asarray(v[f, :, :, channel])                      # (16, 60, 59) uint8

    def batch(self, traps, frame_rows, channel: int) -> np.ndarray:
        return np.stack([self.clip(tr, fr, channel) for tr, fr in zip(traps, frame_rows)])


# ------------------------------------------------------------------ the channel choice and intensity map
def laplacian_variance(frame: np.ndarray) -> float:
    """Variance of the 4-neighbour discrete Laplacian on the interior: a focus measure that needs no labels."""
    f = np.asarray(frame, dtype=np.float64)
    lap = f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:] - 4.0 * f[1:-1, 1:-1]
    return float(lap.var())


def frame_list(store: VidStore, traps: list[str]) -> list[tuple[str, int]]:
    return [(tr, i) for tr in sorted(traps) for i in range(len(store.video(tr)))]


def choose_channel(store: VidStore, traps_by_position: dict, heldout: str, *, n: int = CHANNEL_SAMPLE,
                   seed: int = C.SEED) -> dict:
    """Per fold, the channel with the highest median Laplacian variance over n seeded frames of the fold's 6
    training positions (ties go to the lower channel)."""
    train = C.training_positions(heldout)
    if heldout in train or len(train) != len(C.POSITIONS) - 1:
        raise StoreError(f"fold {heldout}: the channel choice must read the 6 training positions only")
    traps = [tr for p in train for tr in traps_by_position[p]]
    if any(trap_position(tr) == heldout for tr in traps):
        raise StoreError(f"fold {heldout}: a held-out trap reached the channel choice")
    frames = frame_list(store, traps)
    pick = np.random.default_rng(seed).choice(len(frames), size=min(n, len(frames)), replace=False)
    lv = np.array([[laplacian_variance(store.video(frames[i][0])[frames[i][1], :, :, c]) for c in range(3)]
                   for i in sorted(pick)])
    med = np.median(lv, axis=0)
    return {"channel": int(np.argmax(med)), "median_laplacian_variance": [float(v) for v in med],
            "n_frames": int(len(pick)), "positions": list(train)}


def channel_moments(store: VidStore, traps: list[str], channel: int) -> tuple[float, float]:
    """Mean and standard deviation of channel / 255 over every pixel of every frame of the given traps."""
    s = ss = cnt = 0.0
    for tr in sorted(traps):
        x = np.asarray(store.video(tr)[:, :, :, channel], dtype=np.float64) / 255.0
        s += float(x.sum()); ss += float((x * x).sum()); cnt += x.size
    mean = s / cnt
    return mean, float(np.sqrt(max(ss / cnt - mean * mean, 0.0)))


def phase_to_unit_np(x: np.ndarray) -> np.ndarray:
    """numpy version of the fixed phase map into [0, 1] (``deadahead.video.training.phase_to_unit``)."""
    from deadahead.video.model import PHASE_HI, PHASE_LO
    lin = np.clip((np.asarray(x, dtype=np.float64) - PHASE_LO) / (PHASE_HI - PHASE_LO), 0, 1)
    return np.log1p(9.0 * lin) / np.log(10.0)


def source_input_moments(store_dir: Path, *, n: int = SOURCE_SAMPLE, seed: int = C.SEED,
                         exclude: tuple = SOURCE_EXCLUDED_MOVIES) -> dict:
    """m_F and s_F: mean and spread of the [0, 1] inputs the quantitative-phase VideoMAE was trained on, over n
    seeded frames of its crop store (frames drawn uniformly over the store's rows, the excluded movie left out)."""
    shards = sorted(p for p in Path(store_dir).glob("*.npy") if p.stem not in exclude)
    lens = np.array([np.load(p, mmap_mode="r").shape[0] for p in shards], dtype=np.int64)
    cum = np.r_[0, np.cumsum(lens)]
    pick = np.sort(np.random.default_rng(seed).choice(int(cum[-1]), size=min(n, int(cum[-1])), replace=False))
    s = ss = cnt = 0.0
    for g in pick:
        k = int(np.searchsorted(cum, g, side="right") - 1)
        x = phase_to_unit_np(np.load(shards[k], mmap_mode="r")[int(g - cum[k])])
        s += float(x.sum()); ss += float((x * x).sum()); cnt += x.size
    mean = s / cnt
    return {"m_F": mean, "s_F": float(np.sqrt(max(ss / cnt - mean * mean, 0.0))), "n_frames": int(len(pick)),
            "n_movies": len(shards), "excluded": list(exclude), "seed": seed}


def to_unit(x_uint8: np.ndarray, m_d: float, s_d: float, m_f: float, s_f: float) -> np.ndarray:
    """The intensity map: x' = clip((x / 255 - m_D) / s_D * s_F + m_F, 0, 1), float32, no inversion."""
    x = np.asarray(x_uint8, dtype=np.float32) / 255.0
    return np.clip((x - m_d) / s_d * s_f + m_f, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ the picture measurements
def otsu_threshold(hist: np.ndarray) -> int:
    """Otsu's threshold on a 256-bin histogram: the k maximising the between-class variance of {<= k, > k}."""
    h = np.asarray(hist, dtype=np.float64)
    w0 = np.cumsum(h)
    w1 = h.sum() - w0
    mu = np.cumsum(h * np.arange(len(h)))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * w0 - mu * h.sum()) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = -1.0
    return int(np.argmax(between[:-1]))


def size_rule(store: VidStore, traps: list[str], channel: int) -> dict:
    """One Otsu threshold and the cell side per fold, from the training positions' frames (no label); the cell side
    holds the smaller share of the pixels (a tie goes to 'below')."""
    hist = np.zeros(256, dtype=np.int64)
    for tr in sorted(traps):
        hist += np.bincount(np.asarray(store.video(tr)[:, :, :, channel]).ravel(), minlength=256)
    thr = otsu_threshold(hist)
    below = float(hist[:thr + 1].sum()) / float(hist.sum())
    return {"threshold": thr, "side": "below" if below <= 0.5 else "above", "share_below": below}


def frame_measurements(video_channel: np.ndarray, threshold: int, side: str) -> np.ndarray:
    """(T, 3): per frame, the trap's mean and standard deviation of brightness (/ 255) and the size proxy (the number
    of pixels on the cell side of the threshold)."""
    v = np.asarray(video_channel)
    x = v.astype(np.float64) / 255.0
    cell = (v <= threshold) if side == "below" else (v > threshold)
    T = len(v)
    return np.column_stack([x.reshape(T, -1).mean(1), x.reshape(T, -1).std(1), cell.reshape(T, -1).sum(1)])


def picture_features(keys: pd.DataFrame, store: VidStore, channel: int, threshold: int, side: str) -> pd.DataFrame:
    """At each landmark t, from frames at or before t: the measurements at t and their 1-h trends (value at t minus
    value at max(t - 12, 0))."""
    out = []
    for trap, g in keys.groupby("unit", sort=True):
        m = frame_measurements(store.video(trap)[:, :, :, channel], threshold, side)
        t = g["t"].to_numpy()
        tb = np.maximum(t - C.TREND_FRAMES, 0)
        out.append(pd.DataFrame({"unit": trap, "t": t, "pic_mean": m[t, 0], "pic_std": m[t, 1], "pic_size": m[t, 2],
                                 "pic_mean_trend": m[t, 0] - m[tb, 0], "pic_std_trend": m[t, 1] - m[tb, 1],
                                 "pic_size_trend": m[t, 2] - m[tb, 2]}, index=g.index))
    return pd.concat(out).loc[keys.index] if out else pd.DataFrame(columns=["unit", "t", *PIC_SIZE])


# ------------------------------------------------------------------ the row files
def label_efficiency_positions(heldout: str) -> tuple[str, ...]:
    """3 of the 6 training positions, drawn with default_rng(2601562 + fold index), the same for both starts."""
    train = list(C.training_positions(heldout))
    rng = np.random.default_rng(C.SEED + C.fold_index(heldout))
    pick = rng.choice(len(train), size=C.LABEL_EFFICIENCY_POSITIONS, replace=False)
    return tuple(train[i] for i in sorted(pick))


def row_file_positions(kind: str, heldout: str) -> tuple[str, ...]:
    if kind == "final":
        return C.training_positions(heldout)
    if kind == "thr":
        return C.threshold_run_positions(heldout)
    if kind == "le":
        return label_efficiency_positions(heldout)
    raise StoreError(f"unknown row-file kind {kind!r}")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(df: pd.DataFrame, path: Path) -> dict:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return {"file": path.name, "rows": int(len(df)), "sha256": sha256_file(path),
            "positions": sorted(df["position"].unique().tolist()) if "position" in df else None}


def write_row_files(rows: pd.DataFrame, out_dir: Path) -> dict:
    """Per fold p: final__<p> (the 6 training positions; y = the 30-min label, determined rows only), thr__<p> (the 5
    positions other than the threshold position), le__<p> (3 positions), surv__<p> and survthr__<p> (event and
    last-observed offsets of the 6, or the 5, training positions), thrpos_keys__<p> and heldout_keys__<p> (keys
    only), heldout_labels__<p> (the scorer only); plus all_keys, the label features and the clip index. Returns
    the counts and sha256 of every file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = rows[["unit", "position", "t"]].reset_index(drop=True)
    feats = rows[["unit", "position", "t", *L.DETECDIV_FEATURES]].reset_index(drop=True)
    rec = {"all_keys": _write(keys, out_dir / "all_keys.parquet"),
           "label_features": _write(feats, out_dir / "label_features.parquet"),     # inputs at or before t
           "clips": _write(clip_index(keys).assign(position=keys["position"]), out_dir / "clips.parquet"),
           "folds": {}}
    det = rows[rows[f"y_{C.PRIMARY_H}"].notna()].copy()
    det["y"] = det[f"y_{C.PRIMARY_H}"].astype(int)
    for p in C.POSITIONS:
        f = {}
        for kind in ("final", "thr", "le"):
            pos = row_file_positions(kind, p)
            if p in pos:
                raise StoreError(f"fold {p}: the held-out position is in the {kind} rows")
            df = det[det["position"].isin(pos)][["unit", "position", "t", "y"]].reset_index(drop=True)
            r = _write(df, out_dir / f"{kind}__{p}.parquet")
            back = pd.read_parquet(out_dir / f"{kind}__{p}.parquet")
            if sorted(back["position"].unique()) != sorted(pos):
                raise StoreError(f"{kind}__{p}: positions {sorted(back['position'].unique())} != {sorted(pos)}")
            f[kind] = {**r, "positives": int(df["y"].sum())}
        thr = C.threshold_position(p)
        for kind, pos in (("surv", C.training_positions(p)), ("survthr", C.threshold_run_positions(p))):
            sv = rows[rows["position"].isin(pos)][["unit", "position", "t", "ev_off", "obs_off", "nb_off"]]
            f[kind] = _write(sv.reset_index(drop=True), out_dir / f"{kind}__{p}.parquet")
        f["thrpos_keys"] = _write(keys[keys["position"] == thr].reset_index(drop=True),
                                  out_dir / f"thrpos_keys__{p}.parquet")
        f["heldout_keys"] = _write(keys[keys["position"] == p].reset_index(drop=True),
                                   out_dir / f"heldout_keys__{p}.parquet")
        lab = rows[rows["position"] == p][["unit", "position", "t", *[f"y_{h}" for h in C.HORIZONS]]]
        f["heldout_labels"] = {k: v for k, v in _write(lab.reset_index(drop=True),
                                                        out_dir / f"heldout_labels__{p}.parquet").items()}
        f["threshold_position"] = thr
        rec["folds"][p] = f
    return rec


def read_row_file(rows_dir: Path, kind: str, heldout: str) -> pd.DataFrame:
    """One row file of a fold, checked to hold exactly the positions of its kind."""
    path = Path(rows_dir) / f"{kind}__{heldout}.parquet"
    df = pd.read_parquet(path)
    pos = sorted(row_file_positions(kind, heldout))
    if sorted(df["position"].unique()) != pos:
        raise StoreError(f"{path.name}: positions {sorted(df['position'].unique())} are not {pos}")
    return df


# ------------------------------------------------------------------ per-fold inputs
def traps_by_position(series: dict) -> dict:
    """{position: sorted traps}, every trap of the archive (with or without a cell)."""
    return {p: sorted(t for t in series if trap_position(t) == p) for p in C.POSITIONS}


def fold_inputs(store: VidStore, by_pos: dict) -> dict:
    """Per held-out position: the channel choice, the training positions' channel moments (m_D, s_D) and the size
    rule, all from the 6 training positions' pixels."""
    out = {}
    for p in C.POSITIONS:
        ch = choose_channel(store, by_pos, p)
        traps = [tr for q in C.training_positions(p) for tr in by_pos[q]]
        m_d, s_d = channel_moments(store, traps, ch["channel"])
        out[p] = {**ch, "m_D": m_d, "s_D": s_d, "size_rule": size_rule(store, traps, ch["channel"])}
    return out


def size_counts(keys: pd.DataFrame, store: VidStore, channel: int, threshold: int, side: str) -> pd.DataFrame:
    """The size proxy of ``picture_features`` alone, as integer pixel counts: unit, t, pic_size, pic_size_trend."""
    out = []
    for trap, g in keys.groupby("unit", sort=True):
        v = np.asarray(store.video(trap)[:, :, :, channel])
        cell = (v <= threshold) if side == "below" else (v > threshold)
        n = cell.reshape(len(v), -1).sum(1).astype(np.int64)
        t = g["t"].to_numpy()
        tb = np.maximum(t - C.TREND_FRAMES, 0)
        out.append(pd.DataFrame({"unit": trap, "t": t, "pic_size": n[t], "pic_size_trend": n[t] - n[tb]},
                                index=g.index))
    return pd.concat(out).loc[keys.index]


def picture_tables(rows: pd.DataFrame, store: VidStore, folds: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The picture measurements of every landmark in two compact tables: the brightness statistics per channel
    used by a fold (unit, t, channel, ...) and the size proxy per fold as integer pixel counts (unit, t, fold, ...).
    ``fold_pictures`` turns them back into each fold's ``picture_features``, value for value."""
    keys = rows[["unit", "position", "t"]]
    bright, size = [], []
    for ch in sorted({f["channel"] for f in folds.values()}):
        pic = picture_features(keys, store, ch, 0, "below")
        bright.append(pic[["unit", "t", *PIC_BRIGHT]].reset_index(drop=True).assign(channel=np.int8(ch)))
    for p in C.POSITIONS:
        f = folds[p]
        s = size_counts(keys, store, f["channel"], f["size_rule"]["threshold"], f["size_rule"]["side"])
        s = s.reset_index(drop=True)
        for c in ("pic_size", "pic_size_trend"):
            if np.abs(s[c].to_numpy()).max() >= 2 ** 31:
                raise StoreError(f"{c} does not fit in int32")
            s[c] = s[c].astype(np.int32)
        size.append(s.assign(fold=p))
    bright = pd.concat(bright, ignore_index=True)[["unit", "t", "channel", *PIC_BRIGHT]]
    size = pd.concat(size, ignore_index=True)[["unit", "t", "fold", "pic_size", "pic_size_trend"]]
    return bright, size


def fold_pictures(bright: pd.DataFrame, size: pd.DataFrame, folds: dict, rows: pd.DataFrame) -> dict:
    """{held-out position: that fold's picture measurements (float64), aligned to ``rows`` by index}."""
    out = {}
    base = rows[["unit", "t"]].reset_index()
    for p in C.POSITIONS:
        b = bright[bright["channel"] == folds[p]["channel"]].drop(columns="channel")
        s = size[size["fold"] == p].drop(columns="fold")
        m = base.merge(b, on=["unit", "t"], how="left", validate="one_to_one").merge(
            s, on=["unit", "t"], how="left", validate="one_to_one")
        if m[list(PIC_SIZE)].isna().any().any():
            raise StoreError(f"fold {p}: a landmark has no picture measurement")
        for c in ("pic_size", "pic_size_trend"):
            m[c] = m[c].astype(np.float64)
        out[p] = m.set_index("index").rename_axis(rows.index.name)
    return out


def write_tables(series: dict, out: Path, store: VidStore | None = None, folds: dict | None = None) -> pd.DataFrame:
    """classes.parquet (trap, frame, class), and with a pixel store and the fold inputs, brightness.parquet and
    size.parquet; returns the landmark rows."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    series_table(series).to_parquet(out / "classes.parquet", index=False)
    rows, _ = build_rows(series)
    if store is not None and folds is not None:
        bright, size = picture_tables(rows, store, folds)
        bright.to_parquet(out / "brightness.parquet", index=False)
        size.to_parquet(out / "size.parquet", index=False)
    return rows


def tables_missing(tables_dir: Path) -> list[str]:
    """The files of ``TABLE_FILES`` that the folder does not hold."""
    return [n for n in TABLE_FILES if not (Path(tables_dir) / n).is_file()]


def load_tables(tables_dir: Path) -> tuple[dict, pd.DataFrame, dict, dict]:
    """(series, landmark rows, fold inputs, {position: picture frame}) from a folder holding the tables of
    ``write_tables`` and the folds.json of ``main``."""
    d = Path(tables_dir)
    series = load_series(d / "classes.parquet")
    rows, _ = build_rows(series)
    folds = json.loads((d / "folds.json").read_text())
    pics = fold_pictures(pd.read_parquet(d / "brightness.parquet"), pd.read_parquet(d / "size.parquet"),
                         folds["folds"], rows)
    return series, rows, folds, pics


def load_series(path: Path) -> dict:
    """{trap: per-frame class array} from a long table (trap, frame, class)."""
    t = pd.read_parquet(path).sort_values(["trap", "frame"], kind="mergesort")
    return {tr: g["class"].to_numpy().astype(int) for tr, g in t.groupby("trap", sort=True)}


def series_table(series: dict) -> pd.DataFrame:
    """The long table (trap, frame, class) of {trap: class array}."""
    parts = [pd.DataFrame({"trap": tr, "frame": np.arange(len(c), dtype=np.int32), "class": c.astype(np.int8)})
             for tr, c in sorted(series.items())]
    return pd.concat(parts, ignore_index=True)


def check_counts(rows: pd.DataFrame, notes: dict, series: dict) -> None:
    """Raise unless the archive gives the expected traps and deaths per position, lifespans and landmarks."""
    traps = {p: sum(trap_position(t) == p for t in series) for p in C.POSITIONS}
    deaths = {p: int(rows.loc[(rows["position"] == p) & rows["has_event"], "unit"].nunique()) for p in C.POSITIONS}
    bad = []
    if traps != C.TRAPS_BY_POSITION:
        bad.append(f"traps per position {traps}, expected {C.TRAPS_BY_POSITION}")
    if deaths != C.DEATHS_BY_POSITION:
        bad.append(f"deaths per position {deaths}, expected {C.DEATHS_BY_POSITION}")
    if notes["n_lifespans"] != C.N_LIFESPANS:
        bad.append(f"{notes['n_lifespans']} lifespans, expected {C.N_LIFESPANS}")
    if len(rows) != C.N_LANDMARKS:
        bad.append(f"{len(rows)} landmarks, expected {C.N_LANDMARKS}")
    if bad:
        raise StoreError("this is not the DetecDiv lifespan archive: " + "; ".join(bad))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, required=True, help="DetecDiv lifespanAnnotation.zip (read in memory)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: <DEADAHEAD_DATA>/yeast)")
    ap.add_argument("--source-store", type=Path, default=None,
                    help="the quantitative-phase crop store, to recompute the source moments m_F, s_F; without it "
                         "they are read from configs/yeast.yaml")
    ap.add_argument("--no-pixels", action="store_true", help="labels and row files only")
    a = ap.parse_args(argv)
    if not zipfile.is_zipfile(a.zip):
        print(f"{a.zip} is not the DetecDiv archive (docs/DATA.md)", file=sys.stderr)
        return 3
    out = a.out or yeast_dir()
    series = load_label_series(a.zip)
    if not series:
        print(f"{a.zip} holds no timeseries file (docs/DATA.md)", file=sys.stderr)
        return 3
    try:
        check_counts(*build_rows(series), series)
    except StoreError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    store = folds = None
    if not a.no_pixels:
        build_vid_store(a.zip, out / "vid_store")
        store = VidStore(out / "vid_store")
        folds = fold_inputs(store, traps_by_position(series))
        if a.source_store is not None:
            moments = source_input_moments(a.source_store)
        else:
            moments = config_source_moments()
        out.mkdir(parents=True, exist_ok=True)
        (out / "folds.json").write_text(json.dumps({"folds": folds, "source_moments": moments}, indent=1,
                                                   default=str) + "\n")
    rows = write_tables(series, out, store, folds)
    rows.to_parquet(out / "rows.parquet", index=False)
    rec = write_row_files(rows, out / "rows")
    (out / "rows" / "row_files.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"{len(rows)} landmarks of {rows['unit'].nunique()} lifespans -> {out}")
    return 0


def config_source_moments() -> dict:
    """m_F and s_F as listed in configs/yeast.yaml."""
    import yaml
    from deadahead.paths import REPO_ROOT
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "yeast.yaml").read_text())
    return {k: float(cfg["source_moments"][k]) for k in ("m_F", "s_F")}


if __name__ == "__main__":
    sys.exit(main())
