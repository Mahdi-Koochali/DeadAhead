"""Threshold-free survival scores of the survival layer (each seed's survival head on its held-out position, the
mean over seeds per position) and of the survival rules (``label_surv``, ``label_pic_surv``, ``pic_bright_surv``,
``pic_size_surv``), on each held-out position.

  * time-dependent AUC at 30 / 60 / 120 / 240 min: Wolbers' estimand with the determined-row IPCW estimator at
    tau = the horizon, the marker the predicted cumulative incidence of death at that horizon;
  * the C-index of the 4-hour risk: the same estimator with the cumulative incidence at 240 min as the marker, over
    the whole follow-up of the position (tau = its last observed time);
  * the integrated Brier score to 240 min: the IPCW Brier score at the four horizons, integrated by the trapezoid
    over exactly (30, 60, 120, 240), weights (15, 45, 90, 60) / 210.

G is the empirical survivor function of the held-out position's known censoring times only, and pairs inside one
lifespan are excluded from every C. The held-out position stands in for the held-out group and the lifespan (one per
trap) for the track.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from deadahead.evaluation import survival_metrics as M

from . import constants as K

DEATH = 1                                   # the one cause
BAR_SURV_MEMBERS = ("label_surv", "pic_bright_surv", "pic_size_surv", "label_pic_surv")
DEFAULTS = {
    "censoring_time": "C = the last observed time of a censored lifespan; the end of the movie for a death",
    "c_index_tau": "the C-index of the 4-hour risk uses tau = the held-out position's last observed time",
    "positivity_flag": "a horizon with G below 0.25 is flagged beside its value, never dropped",
}


def population(rows_p: pd.DataFrame, n_frames: dict) -> dict:
    """One position's landmarks as survival arrays, in minutes after the landmark: ``T`` the death (``ev_off``) or
    the last observed frame (``obs_off``); ``delta`` 1 death, 0 censored; ``C`` the known censoring time; ``track`` the
    lifespan (the trap). ``n_frames``: {trap: frames in its movie}."""
    ev = rows_p["ev_off"].to_numpy(float)
    obs = rows_p["obs_off"].to_numpy(float)
    dead = np.isfinite(ev)
    t = rows_p["t"].to_numpy(float)
    end_of_movie = np.array([n_frames[u] for u in rows_p["unit"]], dtype=float) - 1.0 - t
    T = np.where(dead, ev, obs) * K.FRAME_MIN
    C = np.where(dead, np.maximum(end_of_movie, ev), obs) * K.FRAME_MIN
    if np.any(C < T) or not np.all(np.isfinite(T)):
        raise ValueError("a landmark's censoring time precedes its observed time")
    return {"T": T, "delta": dead.astype(int), "C": C, "track": rows_p["unit"].astype(str).to_numpy(),
            "keys": rows_p[["unit", "t"]].reset_index(drop=True)}


def scores(pop: dict, cif: dict) -> dict:
    """The three threshold-free scores of one prediction on one position. ``cif``: {h: array aligned to ``pop``'s
    rows}."""
    T, d, C, trk = pop["T"], pop["delta"], pop["C"], pop["track"]
    G = M.CensoringSurvivor(C)                        # the held-out position's censoring times only
    pos = M.positivity(G, K.HORIZONS)
    td, bs = {}, {}
    for h in K.HORIZONS:
        td[h] = M.concordance(T, d, C, cif[h], DEATH, float(h), G, track=trk, pair_set="no_same_track")["c_index"]
        bs[h] = M.brier_ipcw(T, d, C, cif[h], DEATH, float(h), G)
    tau_all = float(np.max(T))
    c4 = M.concordance(T, d, C, cif[240], DEATH, tau_all, G, track=trk, pair_set="no_same_track")
    return {"time_dependent_auc": {str(h): float(v) for h, v in td.items()},
            "brier_ipcw": {str(h): float(v) for h, v in bs.items()},
            "ibs_to_240": float(M.ibs(bs)),
            "c_index_of_the_4h_risk": float(c4["c_index"]), "c_index_pairs": int(c4["n_pairs"]),
            "c_index_tau_min": tau_all,
            "positivity_flagged_horizons": pos["not_scorable_horizons"], "G_at_horizon": pos["G_at_horizon"]}


def aligned_cif(pop: dict, frame: pd.DataFrame) -> dict:
    """A survival frame's ``cif_<h>`` columns on the population's keys (unit, t); every key must have a value."""
    m = pop["keys"].merge(frame[["unit", "t", *[f"cif_{h}" for h in K.HORIZONS]]], on=["unit", "t"], how="left",
                          validate="one_to_one")
    out = {h: m[f"cif_{h}"].to_numpy(float) for h in K.HORIZONS}
    if any(not np.all(np.isfinite(v)) for v in out.values()):
        raise ValueError("a survival frame has no finite cumulative incidence for a landmark of the position")
    return out


def _mean(blocks: list[dict]) -> dict | None:
    """The mean over seeds of each number."""
    if not blocks:
        return None
    out = {}
    for k in ("time_dependent_auc", "brier_ipcw"):
        out[k] = {h: float(np.mean([b[k][h] for b in blocks])) for h in blocks[0][k]}
    for k in ("ibs_to_240", "c_index_of_the_4h_risk"):
        out[k] = float(np.mean([b[k] for b in blocks]))
    out["n_averaged"] = len(blocks)
    return out


def _try_scores(pop: dict, cif_of) -> dict | str:
    """The scores, or "not computable: <why>" (a frame without a finite incidence, a one-class fit)."""
    try:
        return scores(pop, cif_of())
    except ValueError as exc:
        return f"not computable: {exc}"


def position_scores(rows: pd.DataFrame, n_frames: dict, p: str, model_frames: dict, bar_surv: dict | None) -> dict:
    """One held-out position: the survival layer per start (``model_frames``: {start: {seed: frame}}, the mean
    over seeds) and the survival rules (``bar_surv``: {"keys": frame, "cif": {rule: {h: array} or None}}, on all of
    p's rows). Only p's rows enter: G, the pairs and the Brier score never see another position."""
    pop = population(rows[rows["position"] == p].reset_index(drop=True), n_frames)
    out = {"n_landmarks": int(len(pop["T"])), "n_lifespans": int(len(set(pop["track"]))),
           "n_death_landmarks": int(pop["delta"].sum()), "survival_layer": {}, "rules": {}}
    for st, by_seed in model_frames.items():
        per = {str(s): _try_scores(pop, lambda fr=fr: aligned_cif(pop, fr)) for s, fr in by_seed.items()}
        out["survival_layer"][st] = {"per_seed": per,
                           "mean_over_seeds": _mean([v for v in per.values() if isinstance(v, dict)])}
    if bar_surv is not None:
        k = bar_surv["keys"].reset_index(drop=True)
        if not k[["unit", "t"]].equals(pop["keys"][["unit", "t"]]):
            raise ValueError(f"{p}: the bar's survival rows are not the position's landmarks in row order")
        for m in BAR_SURV_MEMBERS:
            c = bar_surv["cif"].get(m)
            out["rules"][m] = _try_scores(pop, lambda c=c: c) if c is not None \
                else "not computable: the survival expansion holds one class only"
    return out


def survival_layer_scores(rows: pd.DataFrame, n_frames: dict, model_frames: dict, bar_surv: dict) -> dict:
    """The threshold-free scores for every position, and the macro (the mean over positions) of each row."""
    per = {p: position_scores(rows, n_frames, p, model_frames.get(p, {}), bar_surv.get(p)) for p in K.POSITIONS}
    macro = {}
    for p, blk in per.items():
        for st, v in blk["survival_layer"].items():
            if v["mean_over_seeds"] is not None:
                macro.setdefault(f"survival_layer:{st}", []).append(v["mean_over_seeds"])
        for m, v in blk["rules"].items():
            if isinstance(v, dict):
                macro.setdefault(f"rule:{m}", []).append(v)
    return {"per_position": per,
            "macro_over_positions": {r: {**(_mean(v) or {}), "n_positions": len(v)} for r, v in macro.items()},
            "definitions": __doc__.split("\n\n")[1].strip(), "defaults": DEFAULTS}
