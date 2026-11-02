"""DeepHit (Lee et al., 2018) on DeadAhead's fixed inputs, one held-out group and seed at a time.

The inputs are DeadAhead's: the 1,024 fine-tuned VideoMAE features of the landmark (from the model that held out the
same group, same seed) and the 112 handcrafted history features. The targets are those of DeadAhead's training
labels on the truncated movies: L, the last 30-min bin at risk (0..8), and kappa (0 none, 1 apoptosis, 2 lytic); an
untyped death ends follow-up. Landmarks with L = 0 carry no likelihood term and are not trained on.

One of eight settings (alpha, sigma, learning rate) is chosen on an inner training group (the training group with
the most lytic 30-min rows): each setting trains on the other seven groups with early stopping on the inner loss
(patience 10, at most 100 epochs) and is scored by its mean AUPRC on the inner group over 2 causes x 4 horizons. The
chosen setting is refit on all eight training groups for exactly its best epoch count and predicts the held-out
group's cumulative incidence F_c(b), b = 1..8 bins of 30 min.

    deadahead baseline-deephit --group DU145_bp --seed 2601562              (8 selection fits, then final)
    deadahead baseline-deephit --group DU145_bp --seed 2601562 --item sel3  (one selection fit)

Network, losses and output layout follow the competing-risks DeepHit of pycox (``nll_pmf_cr`` and
``rank_loss_deephit_cr``), written out here in torch. Three departures: each likelihood term is weighted by the row's
window weight, the extra cell's logit is learned (pycox fixes it at 0), and the logarithms are exact (pycox adds 1e-7).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:                     # CPU install: --help works; fitting needs the [video] extra
    torch = nn = None

from deadahead.evaluation.metrics import auprc
from deadahead.evaluation.rows import GROUPS
from deadahead.paths import runs_dir, tables_dir

N_BINS, N_CAUSES = 8, 2
N_OUT = N_CAUSES * N_BINS + 1                           # 17: one cell per (cause, bin) and one for no typed death by 240 min
HIDDEN_SHARED, HIDDEN_CAUSE, DROPOUT = 256, 128, 0.1
BATCH = 512
MAX_EPOCHS, PATIENCE = 100, 10
GRID = tuple((a, s, lr) for a in (0.2, 0.5) for s in (0.1, 1.0) for lr in (1e-4, 1e-3))   # (alpha, sigma, lr), in order
HORIZON_BINS = (1, 2, 4, 8)                              # 30, 60, 120, 240 min (selection AUPRC)
PARAMS = {1136: 717_841, 2416: 1_373_201}                # parameter count by input width

BIN_MIN = 30.0                                           # bin width of the targets, minutes
CAUSES = ["APOPTOSIS", "LYTIC"]
NO_EVENT, APOP, LYTIC = 0, 1, 2
KEY = ("movie_id", "track_id", "t_obs_frame")
SEL_ITEMS = tuple(f"sel{i}" for i in range(1, len(GRID) + 1))
FINAL = "final"
UNTYPED = "censor_at_death"                              # an untyped death ends follow-up
VIDEO_PREFIX = "videomae_"


class CheckpointMismatch(Exception):
    """A checkpoint at the fit's path was written by another fit."""


class CheckpointUnreadable(CheckpointMismatch):
    """A checkpoint torch.load cannot read."""


if torch is not None:
    class DeepHit(nn.Module):
        """Shared Linear(d, 256)-ReLU-Dropout-Linear(256, 256)-ReLU-Dropout; per cause
        Linear(256 + d, 128)-ReLU-Dropout on [shared output || x] (the residual input of Lee et al.);
        Linear(256, 17) on the two cause outputs; logits out."""

        def __init__(self, d: int):
            super().__init__()
            self.shared = nn.Sequential(nn.Linear(d, HIDDEN_SHARED), nn.ReLU(), nn.Dropout(DROPOUT),
                                        nn.Linear(HIDDEN_SHARED, HIDDEN_SHARED), nn.ReLU(), nn.Dropout(DROPOUT))
            self.cause = nn.ModuleList([nn.Sequential(nn.Linear(HIDDEN_SHARED + d, HIDDEN_CAUSE), nn.ReLU(),
                                                      nn.Dropout(DROPOUT)) for _ in range(N_CAUSES)])
            self.out = nn.Linear(N_CAUSES * HIDDEN_CAUSE, N_OUT)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.shared(x)
            z = torch.cat([h, x], dim=1)
            return self.out(torch.cat([net(z) for net in self.cause], dim=1))


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def cif_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """F_c(b | x) = sum_{b' <= b} p(c, b' | x): (n, 2, 8), non-decreasing in b."""
    p = torch.softmax(logits, dim=1)
    return torch.cumsum(p[:, : N_CAUSES * N_BINS].reshape(-1, N_CAUSES, N_BINS), dim=2)


# ------------------------------------------------------------------ targets
def survival_targets(tte_min: np.ndarray, terminal: np.ndarray, bin_min: float,
                     n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """(last bin observed through, 0..n_bins; cause, 0 none, 1 apoptosis, 2 lytic) per landmark, from the time to the
    end of follow-up and how it ended. A death inside the grid is observed through its own bin; a censored landmark
    through the bin before its follow-up ends, capped at the grid."""
    b = np.ceil(np.asarray(tte_min, dtype=float) / bin_min)
    b = np.clip(b, 1, n_bins + 1)
    died = np.isin(terminal, CAUSES)
    inside = died & (b <= n_bins)

    cause = np.full(len(b), NO_EVENT, dtype=np.int64)
    cause[inside & (np.asarray(terminal) == "APOPTOSIS")] = APOP
    cause[inside & (np.asarray(terminal) == "LYTIC")] = LYTIC

    # a death inside the grid is observed through its own bin; everything else through the bin before its follow-up
    # ends, capped at the grid
    last_bin = np.where(inside, b, np.minimum(b - 1, n_bins)).astype(np.int64)
    return np.clip(last_bin, 0, n_bins), cause


def l0_mask(L) -> np.ndarray:
    """True for the landmarks DeepHit trains on (L >= 1). L = 0 landmarks (all none) have no likelihood term and no
    comparable pair; they are in no batch and no weight sum, and are predicted like every held-out landmark."""
    L = np.asarray(L)
    return L >= 1


# ------------------------------------------------------------------ loss
def comparable(L: torch.Tensor, k: torch.Tensor, cause: int) -> torch.Tensor:
    """A_ij for cause c: kappa_i = c and [(kappa_j typed and L_j > b_i) or (kappa_j none and L_j >= b_i)], b_i = L_i
    (pycox ``_pair_rank_mat`` with duration = last bin, event = kappa). A same-bin death is not comparable."""
    bi = L[:, None]
    typed_j = (k[None, :] > 0) & (L[None, :] > bi)
    none_j = (k[None, :] == 0) & (L[None, :] >= bi)
    return ((k[:, None] == cause) & (typed_j | none_j)).to(torch.float64 if L.dtype == torch.float64 else torch.float32)


def nll(logits: torch.Tensor, L: torch.Tensor, k: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """pycox ``nll_pmf_cr`` with each term weighted by ``window_weight`` and normalised by the batch's weight sum.
    Event (L, c): -log p(c, L); none after L bins: -log(1 - sum_c F_c(L)) (L = 8: the extra cell's mass)."""
    logz = torch.logsumexp(logits, dim=1)
    n = logits.shape[0]
    idx = torch.arange(n)
    ev = k > 0
    cell = torch.clamp((k - 1) * N_BINS + (L - 1), min=0)
    log_event = logits[idx, cell] - logz
    b = torch.arange(1, N_BINS + 1)
    surv_cells = torch.cat([(b[None, :] > L[:, None]).repeat(1, N_CAUSES), torch.ones(n, 1, dtype=torch.bool)], dim=1)
    log_surv = torch.logsumexp(logits.masked_fill(~surv_cells, float("-inf")), dim=1) - logz
    term = torch.where(ev, -log_event, -log_surv)
    return (w * term).sum() / w.sum()


def rank_loss(logits: torch.Tensor, L: torch.Tensor, k: torch.Tensor, sigma: float) -> torch.Tensor:
    """pycox ``rank_loss_deephit_cr``: per cause, (1/n) sum_i (1/n) sum_j A_ij exp(-R_ij / sigma), R_ij =
    F_c(b_i | x_i) - F_c(b_i | x_j); summed over the two causes."""
    F = cif_from_logits(logits)                                   # (n, 2, 8)
    bi = torch.clamp(L - 1, min=0)
    total = logits.new_zeros(())
    for c in (1, 2):
        Fc = F[:, c - 1, :]                                       # (n, 8)
        at_bi = Fc[:, bi]                                         # [j, i] = F_c(b_i | x_j)
        R = torch.diagonal(at_bi)[:, None] - at_bi.T              # [i, j] = F_c(b_i|x_i) - F_c(b_i|x_j)
        A = comparable(L, k, c).to(logits.dtype)
        total = total + (A * torch.exp(-R / sigma)).mean(dim=1).mean()
    return total


def loss(logits, L, k, w, alpha: float, sigma: float) -> torch.Tensor:
    """alpha * NLL + (1 - alpha) * rank (pycox ``DeepHitLoss.forward``)."""
    return alpha * nll(logits, L, k, w) + (1.0 - alpha) * rank_loss(logits, L, k, sigma)


# ------------------------------------------------------------------ standardisation and imputation
def fit_standardiser(X: np.ndarray) -> dict:
    """Per-feature median (imputation of missing values), mean and SD, fitted on the fit's own training rows only.
    A column with no finite value is imputed with 0; a column with SD 0 is divided by 1."""
    X = np.asarray(X, dtype=np.float64)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xi = np.where(np.isnan(X), med[None, :], X)
    mu, sd = Xi.mean(axis=0), Xi.std(axis=0)
    return {"median": med, "mean": mu, "sd": np.where(sd > 0, sd, 1.0)}


def apply_standardiser(X: np.ndarray, st: dict) -> torch.Tensor:
    X = np.asarray(X, dtype=np.float64)
    Xi = np.where(np.isnan(X), st["median"][None, :], X)
    return torch.as_tensor((Xi - st["mean"]) / st["sd"], dtype=torch.float32)


# ------------------------------------------------------------------ training with checkpoints
def _batches(n: int, size: int = BATCH):
    return [np.arange(s, min(s + size, n)) for s in range(0, n, size)]


def inner_loss(model: DeepHit, X: torch.Tensor, L, k, w, alpha: float, sigma: float) -> float:
    """The early-stopping loss on the inner group: fixed batches of 512 in key order (the caller passes key order),
    the same reduction, averaged over batches; eval mode (no dropout)."""
    model.eval()
    with torch.no_grad():
        vals = [float(loss(model(X[b]), L[b], k[b], w[b], alpha, sigma)) for b in _batches(X.shape[0])]
    model.train()
    return float(np.mean(vals))


def train(X_tr, L_tr, k_tr, w_tr, setting: tuple, seed: int, X_in=None, L_in=None, k_in=None, w_in=None,
          epochs: int | None = None, ckpt: Path | None = None, stop_after: int | None = None,
          run_info: dict | None = None) -> dict:
    """One fit. Selection (``epochs`` None): at most 100 epochs with patience 10 on the inner loss; e* = the epoch
    with the lowest inner loss (the first on a tie; the untrained initialisation is never a candidate); the setting is
    no-improvement when no epoch is below the initialisation's loss. Final (``epochs`` = e*): exactly that many epochs.
    L = 0 landmarks are removed here. A checkpoint after every epoch (model, optimiser, torch and numpy random states,
    epoch, losses, best state) lets a restarted fit resume. ``stop_after`` stops after that many epochs (tests);
    ``run_info`` is recorded in the checkpoint's list of (re)starts."""
    alpha, sigma, lr = setting
    sel = epochs is None
    n_target = MAX_EPOCHS if sel else int(epochs)
    ident = fit_identity(X_tr, L_tr, k_tr, w_tr, setting, seed, n_target, (X_in, L_in, k_in, w_in) if sel else None)
    keep = l0_mask(L_tr)
    st = fit_standardiser(np.asarray(X_tr)[keep])                 # fitted on the rows the fit trains on
    X = apply_standardiser(np.asarray(X_tr)[keep], st)
    L = torch.as_tensor(np.asarray(L_tr)[keep], dtype=torch.long)
    k = torch.as_tensor(np.asarray(k_tr)[keep], dtype=torch.long)
    w = torch.as_tensor(np.asarray(w_tr, dtype=np.float64)[keep], dtype=torch.float32)
    if sel:
        ki = l0_mask(L_in)
        Xi = apply_standardiser(np.asarray(X_in)[ki], st)
        Li = torch.as_tensor(np.asarray(L_in)[ki], dtype=torch.long)
        kin = torch.as_tensor(np.asarray(k_in)[ki], dtype=torch.long)
        wi = torch.as_tensor(np.asarray(w_in, dtype=np.float64)[ki], dtype=torch.float32)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = DeepHit(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng([seed, 7])
    hist = {"train": [], "inner": [], "init_inner": None, "best_epoch": None, "best_inner": None}
    best_state, epoch, segments = None, 0, []
    if ckpt is not None and Path(ckpt).is_file():
        try:
            cp = torch.load(ckpt, weights_only=False)
        except Exception as e:  # noqa: BLE001 -- a truncated or corrupt file recurs on every retry
            raise CheckpointUnreadable(f"{ckpt}: unreadable ({type(e).__name__}: {e}); not resumed") from e
        theirs = cp.get("identity") or {}
        diff = sorted(k for k in ident if theirs.get(k) != ident[k])
        if diff:
            raise CheckpointMismatch(f"{ckpt}: written by another fit ({', '.join(diff)} differ); not resumed")
        model.load_state_dict(cp["model"])
        opt.load_state_dict(cp["opt"])
        torch.set_rng_state(cp["torch_rng"])
        rng.bit_generator.state = cp["np_rng"]
        hist, epoch, best_state = cp["hist"], cp["epoch"], cp["best_state"]
        segments = list(cp.get("segments") or [])
    elif sel:
        hist["init_inner"] = inner_loss(model, Xi, Li, kin, wi, alpha, sigma)
    segments.append({"first_epoch": epoch + 1, "last_epoch": epoch, **(run_info or {})})
    while epoch < n_target:
        if sel and hist["best_epoch"] is not None and epoch - hist["best_epoch"] >= PATIENCE:
            break
        order = rng.permutation(X.shape[0])
        tl = []
        for b in _batches(X.shape[0]):
            ib = order[b]
            opt.zero_grad()
            lv = loss(model(X[ib]), L[ib], k[ib], w[ib], alpha, sigma)
            if not torch.isfinite(lv):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
            lv.backward()
            opt.step()
            tl.append(float(lv))
        epoch += 1
        hist["train"].append(float(np.mean(tl)))
        if sel:
            v = inner_loss(model, Xi, Li, kin, wi, alpha, sigma)
            hist["inner"].append(v)
            if hist["best_inner"] is None or v < hist["best_inner"]:
                hist["best_inner"], hist["best_epoch"] = v, epoch
                best_state = {kk: t.clone() for kk, t in model.state_dict().items()}
        segments[-1]["last_epoch"] = epoch
        if ckpt is not None:
            tmp = Path(ckpt).with_name(f"{Path(ckpt).name}.{os.getpid()}.tmp")   # written whole, then renamed
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "torch_rng": torch.get_rng_state(),
                        "np_rng": rng.bit_generator.state, "hist": hist, "epoch": epoch, "best_state": best_state,
                        "identity": ident, "segments": segments}, tmp)
            os.replace(tmp, ckpt)
        if stop_after is not None and epoch >= stop_after:
            return {"interrupted": True, "epoch": epoch, "segments": segments}
    out = {"setting": setting, "seed": seed, "epochs_run": epoch, "hist": hist, "standardiser": st,
           "n_l0_removed": int((~keep).sum()), "model": model, "segments": segments}
    if sel:
        out["e_star"] = hist["best_epoch"]
        out["no_improvement"] = hist["best_inner"] is None or not (hist["best_inner"] < hist["init_inner"])
        out["cap_flag"] = hist["best_epoch"] == MAX_EPOCHS
        out["best_state"] = best_state
    return out


def fit_identity(X_tr, L_tr, k_tr, w_tr, setting: tuple, seed: int, n_target: int, inner: tuple | None = None) -> dict:
    """What a checkpoint must match to be resumed: the setting, the seed, the mode (``inner`` = (X_in, L_in, k_in,
    w_in) for a selection fit, None for a final fit), the target epoch count and a sha256 over the arrays."""
    sel = inner is not None
    h = hashlib.sha256()
    arrs = [(X_tr, np.float64), (L_tr, np.int64), (k_tr, np.int64), (w_tr, np.float64)]
    if sel:
        arrs += list(zip(inner, (np.float64, np.int64, np.int64, np.float64)))
    for a, dt in arrs:
        a = np.ascontiguousarray(np.asarray(a, dtype=dt))
        h.update(repr(a.shape).encode())
        h.update(a.tobytes())
    return {"setting": [float(x) for x in setting], "seed": int(seed), "mode": "selection" if sel else "final",
            "n_target": int(n_target), "inputs_sha256": h.hexdigest()}


def predict(model: DeepHit, X, st: dict, state: dict | None = None) -> dict:
    """F_c(b) for c in (APOPTOSIS, LYTIC), b = 1..8, and the extra cell's mass, for every row (eval mode)."""
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits = model(apply_standardiser(X, st))
        F = cif_from_logits(logits).numpy().astype(np.float64)
        extra = torch.softmax(logits, dim=1)[:, -1].numpy().astype(np.float64)
    return {"F_APOPTOSIS": F[:, 0, :], "F_LYTIC": F[:, 1, :], "extra_mass": extra}


def inner_auprc(pred: dict, L_in, k_in) -> float:
    """Mean AUPRC over 2 causes x 4 horizons on the inner group's determined rows: event of c by bin h iff kappa = c
    and L <= h; determined iff a death by h or L >= h. Cells without an event are left out of the mean."""
    L, k = np.asarray(L_in), np.asarray(k_in)
    vals = []
    for ci, cname in ((1, "F_APOPTOSIS"), (2, "F_LYTIC")):
        for hb in HORIZON_BINS:
            det = ((k > 0) & (L <= hb)) | (L >= hb)
            y = ((k == ci) & (L <= hb))[det]
            vals.append(auprc(y, pred[cname][det, hb - 1]) if y.any() else float("nan"))
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def constant_output_flags(pred: dict) -> dict:
    """Per cause, whether F_c(8) is the same for every predicted row."""
    return {c: bool(np.ptp(np.asarray(pred[f"F_{c}"])[:, N_BINS - 1]) == 0) for c in ("APOPTOSIS", "LYTIC")}


def choose_setting(results: list[dict]) -> int | None:
    """The eligible setting (not no-improvement, finite inner AUPRC) with the highest inner mean AUPRC; the first in
    grid order on a tie; None when no setting is eligible."""
    best, best_i = None, None
    for i, r in enumerate(results):
        if r.get("no_improvement") or not np.isfinite(r.get("inner_auprc", np.nan)):
            continue

        if best is None or r["inner_auprc"] > best:
            best, best_i = r["inner_auprc"], i
    return best_i


# ------------------------------------------------------------------ the fold's arrays
def key_order(movie, track, t_obs) -> np.ndarray:
    """Row positions sorted by (movie_id, track_id, t_obs_frame): the inner group's fixed batch order."""
    movie, track, t_obs = np.asarray(movie).astype(str), np.asarray(track).astype(str), np.asarray(t_obs)
    return np.asarray(sorted(range(len(movie)), key=lambda i: (movie[i], track[i], int(t_obs[i]))), dtype=np.int64)


def fold_split(bundle: dict, fold: str, inner: str) -> dict:
    """Training rows of the selection fits (the 7 groups other than the held-out and the inner group), the inner
    group's rows in key order, the final fit's training rows (all 8 training groups) and the held-out rows."""
    g = np.asarray(bundle["group"]).astype(str)
    if inner == fold or inner not in set(g) or fold not in set(g):
        raise ValueError(f"the inner group {inner!r} is not a training group of fold {fold!r}")
    held, inn = g == fold, g == inner
    pos_inner = np.flatnonzero(inn)
    o = key_order(np.asarray(bundle["movie_id"])[pos_inner], np.asarray(bundle["track_id"])[pos_inner],
                  np.asarray(bundle["t_obs_frame"])[pos_inner])
    return {"sel_train": np.flatnonzero(~held & ~inn), "inner": pos_inner[o], "final_train": np.flatnonzero(~held),
            "held": np.flatnonzero(held)}


def bundle_from(frame_tr, frame_te, cols, lm, movies, fold: str, untyped: str) -> dict:
    """One row per landmark of the fold's training and held-out frames: X (float32), L and kappa (training landmarks
    only, from ``survival_targets`` on the truncated movies; 0 for the held-out ones), window weight, group, keys."""
    import pandas as pd

    from deadahead.survival.frames import CUT_FRAME, reference_inputs
    fx = pd.concat([frame_tr, frame_te], ignore_index=True).drop_duplicates(subset=list(KEY)).reset_index(drop=True)
    sel = lm.set_index(list(KEY)).reindex(pd.MultiIndex.from_frame(fx[list(KEY)])).reset_index()
    L = np.zeros(len(fx), dtype=np.int64)
    kap = np.zeros(len(fx), dtype=np.int64)
    trm = (fx["group"] != fold).to_numpy()
    tte, term = reference_inputs(sel[trm], untyped, movies, CUT_FRAME)
    last_bin, cause = survival_targets(tte, term, BIN_MIN, N_BINS)
    L[trm], kap[trm] = np.asarray(last_bin), np.asarray(cause)
    return {"X": fx[cols].to_numpy(dtype=np.float32), "L": L, "k": kap,
            "w": fx["window_weight"].to_numpy(dtype=np.float64), "group": fx["group"].to_numpy().astype(str),
            **{c: fx[c].to_numpy() for c in KEY}}


def build_bundle(fold: str, tables: dict, hist, rows, feats: np.ndarray, cfg: dict, cfg_w: dict) -> dict:
    """The fold's arrays on DeadAhead's inputs: every landmark of the primary panel with its 1,024 VideoMAE features
    (``feats`` row i belongs to ``rows`` row i) followed by the 112 handcrafted features."""
    import pandas as pd

    from deadahead.survival.frames import (
        CUT_FRAME,
        apply_view,
        assert_inside_cut,
        build_data,
        input_columns,
        join_store,
        landmark_table,
        panel_view,
        panel_windows,
        split_fold,
        store_positions,
    )
    data, phys = build_data(tables["windows"], tables["events"], tables["movies"], hist, cfg, cfg_w, "primary")
    lm = landmark_table(panel_windows(tables["windows"], "primary"))
    tr0, te = split_fold(data, fold)
    store_positions(data, rows)                              # every landmark has a feature row
    assert_inside_cut(apply_view(tr0, panel_view(lm, UNTYPED, tables["movies"], CUT_FRAME)), tables["movies"])
    frame = pd.concat([tr0, te], ignore_index=True).drop_duplicates(subset=list(KEY)).reset_index(drop=True)
    fx, scols = join_store(frame, rows, feats, VIDEO_PREFIX)
    cols = input_columns("deadahead", scols, phys)
    return bundle_from(fx, fx.iloc[:0], cols, lm, tables["movies"], fold, UNTYPED)


def inner_group(fold: str, windows_parquet: Path) -> str:
    """The selection group of the fold: the training group with the most lytic 30-min rows (as for VideoMAE)."""
    from deadahead.video.finetune import inner_validation_group, label_table
    return inner_validation_group(label_table(windows_parquet), [g for g in GROUPS if g != fold])


# ------------------------------------------------------------------ work items
def _clean(o):
    """JSON-safe: nan / inf to None, numpy scalars to Python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _write_json(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_clean(rec), indent=1) + "\n")
    tmp.replace(p)


def unit_name(fold: str, seed: int) -> str:
    return f"deephit__{fold}__s{seed}"


def default_out_dir() -> Path:
    return runs_dir() / "baselines" / "deephit"


def sel_path(out: Path, unit: str, item: str) -> Path:
    return Path(out) / "sel" / f"{unit}__{item}.json"


def pred_path(out: Path, unit: str) -> Path:
    return Path(out) / "pred" / f"{unit}.npz"


def record_path(out: Path, unit: str) -> Path:
    """The record of the final fit, written last: a unit is finished when this file exists."""
    return Path(out) / f"{unit}.json"


def run_selection(bundle: dict, fold: str, seed: int, item: str, inner: str, out: Path,
                  stop_after: int | None = None) -> dict | None:
    """Selection fit ``item`` (sel1..sel8): train on the seven groups other than the held-out and the inner group,
    stop early on the inner group, score the best state's inner AUPRC; writes sel/<unit>__<item>.json."""
    unit = unit_name(fold, seed)
    i = SEL_ITEMS.index(item)
    sp = fold_split(bundle, fold, inner)
    X, L, k, w = (np.asarray(bundle[c]) for c in ("X", "L", "k", "w"))
    t0 = time.monotonic()
    ck = Path(out) / "ckpt" / f"{unit}__{item}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    tr, inn = sp["sel_train"], sp["inner"]
    res = train(X[tr], L[tr], k[tr], w[tr], GRID[i], seed, X_in=X[inn], L_in=L[inn], k_in=k[inn], w_in=w[inn],
                ckpt=ck, stop_after=stop_after)
    if res.get("interrupted"):
        return None
    score = float("nan")
    if res["best_state"] is not None:
        pred = predict(res["model"], X[inn], res["standardiser"], state=res["best_state"])
        score = inner_auprc(pred, L[inn], k[inn])
    rec = {"unit": unit, "item": item, "setting_index": i + 1,
           "setting": {"alpha": GRID[i][0], "sigma": GRID[i][1], "lr": GRID[i][2]}, "seed": seed,
           "inner_group": inner, "rows": {"train": int(len(tr)), "inner": int(len(inn))},
           "n_l0_removed_train": int(res["n_l0_removed"]), "n_l0_removed_inner": int((~l0_mask(L[inn])).sum()),
           "e_star": res["e_star"], "no_improvement": bool(res["no_improvement"]), "cap_flag": bool(res["cap_flag"]),
           "init_inner_loss": res["hist"]["init_inner"], "train_loss": res["hist"]["train"],
           "inner_loss": res["hist"]["inner"], "inner_auprc": score, "epochs_run": res["epochs_run"],
           "wall_s": round(time.monotonic() - t0, 3)}
    _write_json(sel_path(out, unit, item), rec)
    return rec


def run_final(bundle: dict, fold: str, seed: int, inner: str, out: Path) -> dict:
    """Read the eight selection records, choose the setting, refit on all eight training groups for e* epochs and
    predict the held-out group; writes pred/<unit>.npz and <unit>.json."""
    unit = unit_name(fold, seed)
    sel = []
    for item in SEL_ITEMS:
        p = sel_path(out, unit, item)
        if not p.is_file():
            raise FileNotFoundError(f"{p}: the final fit needs all eight selection fits")
        sel.append(json.loads(p.read_text()))
    results = [{"no_improvement": r["no_improvement"],
                "inner_auprc": float("nan") if r["inner_auprc"] is None else r["inner_auprc"]} for r in sel]
    chosen = choose_setting(results)
    base = {"unit": unit, "fold": fold, "seed": seed, "inner_group": inner,
            "settings": [{k: r.get(k) for k in ("setting_index", "setting", "e_star", "no_improvement", "cap_flag",
                                                "inner_auprc")} for r in sel]}
    if chosen is None:
        rec = {**base, "status": "not run", "reason": "no eligible setting: all eight are no-improvement"}
        _write_json(record_path(out, unit), rec)
        return rec
    t0 = time.monotonic()
    e_star = int(sel[chosen]["e_star"])
    sp = fold_split(bundle, fold, inner)
    X, L, k, w = (np.asarray(bundle[c]) for c in ("X", "L", "k", "w"))
    tr, held = sp["final_train"], sp["held"]
    ck = Path(out) / "ckpt" / f"{unit}__final.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    res = train(X[tr], L[tr], k[tr], w[tr], GRID[chosen], seed, epochs=e_star, ckpt=ck)
    pred = predict(res["model"], X[held], res["standardiser"])
    if not all(np.isfinite(pred[c]).all() for c in ("F_APOPTOSIS", "F_LYTIC", "extra_mass")):
        raise FloatingPointError("a non-finite prediction")
    pp = pred_path(out, unit)
    pp.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, movie_id=np.asarray(bundle["movie_id"])[held].astype(str),
             track_id=np.asarray(bundle["track_id"])[held].astype(str),
             t_obs_frame=np.asarray(bundle["t_obs_frame"])[held].astype(np.int64), **pred)
    tmp = pp.with_name(f"{pp.name}.{os.getpid()}.tmp")           # written whole, then renamed
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, pp)
    rec = {**base, "status": "done", "chosen_setting_index": chosen + 1, "e_star": e_star,
           "rows": {"train": int(len(tr)), "held_out": int(len(held))}, "n_l0_removed_train": res["n_l0_removed"],
           "epochs_run": res["epochs_run"], "train_loss": res["hist"]["train"],
           "flags": {"constant_output": constant_output_flags(pred),
                     "no_improvement_settings": [s["setting_index"] for s in base["settings"] if s["no_improvement"]],
                     "cap_settings": [s["setting_index"] for s in base["settings"] if s["cap_flag"]]},
           "prediction": str(pp), "wall_s": round(time.monotonic() - t0, 3)}
    _write_json(record_path(out, unit), rec)
    return rec


def set_threads(n: int) -> None:
    """Pin the BLAS and torch thread counts (set before the first fit)."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(int(n))
    torch.set_num_threads(int(n))


def load_inputs(fold: str, seed: int, features: Path | None = None, rows: Path | None = None,
                tables_dir_: Path | None = None, history: Path | None = None) -> tuple[dict, str]:
    """The fold's arrays and its inner group, from the tables, the history features and the VideoMAE features of the
    model that held out ``fold`` with ``seed``."""
    import pandas as pd

    from deadahead.features.history import history_features
    from deadahead.survival.config import head_config, windows_config
    from deadahead.survival.frames import load_tables
    from deadahead.survival.train import default_features, default_rows, load_features
    d = Path(tables_dir_) if tables_dir_ is not None else tables_dir()
    tables = load_tables(d, names=("windows", "events", "movies"))
    if history is not None:
        hist = pd.read_parquet(history)
    else:
        hist = history_features(int(windows_config()["landmark"]["history_frames_W"]),
                                (d / "cells.parquet") if tables_dir_ is not None else None)
    rows_df = pd.read_parquet(rows or default_rows())
    feats = load_features(features or default_features(fold, seed), len(rows_df))
    bundle = build_bundle(fold, tables, hist, rows_df, feats, head_config(), windows_config())
    return bundle, inner_group(fold, d / "windows.parquet")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DeepHit on DeadAhead's inputs for one held-out group and seed.")
    ap.add_argument("--group", "--fold", dest="group", required=True, choices=GROUPS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--item", default="all", help="sel1..sel8, final, or all (the default: eight selection fits, "
                                                  "then the final fit; finished fits are skipped)")
    ap.add_argument("--features", type=Path, help="VideoMAE features (default runs/videomae/features/<group>__s<seed>.npy)")
    ap.add_argument("--rows", type=Path, help="row index of the features (default runs/videomae/features/rows.parquet)")
    ap.add_argument("--tables-dir", type=Path, help="folder with windows, events, movies and cells .parquet")
    ap.add_argument("--history", type=Path, help="precomputed history features (.parquet); default: built and cached")
    ap.add_argument("--out-dir", type=Path, default=None, help="default runs/baselines/deephit")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    if torch is None:
        ap.error('PyTorch is not installed: pip install -e ".[video]"')
    items = [*SEL_ITEMS, FINAL] if a.item == "all" else [a.item]
    if any(i not in (*SEL_ITEMS, FINAL) for i in items):
        ap.error(f"--item: {a.item!r} is not sel1..sel8, final or all")
    set_threads(a.threads)
    out = a.out_dir or default_out_dir()
    unit = unit_name(a.group, a.seed)
    todo = [i for i in items if not (sel_path(out, unit, i) if i != FINAL else record_path(out, unit)).is_file()]
    if not todo:
        print(f"{unit}: done")
        return 0
    bundle, inner = load_inputs(a.group, a.seed, a.features, a.rows, a.tables_dir, a.history)
    for item in todo:
        if item == FINAL:
            rec = run_final(bundle, a.group, a.seed, inner, out)
            print(f"{unit} final: {rec['status']}" + (f", setting {rec['chosen_setting_index']}, e* {rec['e_star']}"
                                                       if rec["status"] == "done" else ""), flush=True)
        else:
            rec = run_selection(bundle, a.group, a.seed, item, inner, out)
            print(f"{unit} {item}: e* {rec['e_star']}, inner AUPRC {rec['inner_auprc']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
