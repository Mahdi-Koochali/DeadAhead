"""Ranking metrics for one cause at one horizon (AUPRC, AUROC), numpy only."""

from __future__ import annotations

import numpy as np


def auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Average precision (step-wise AUPRC): AP = sum_k (R_k - R_{k-1}) * P_k over distinct score thresholds.

    Rows that share a score form one threshold, so the value does not depend on row order (a constant score gives
    the prevalence). Returns nan when there is no positive.
    """
    y_true = np.asarray(y_true).astype(bool)
    score = np.asarray(score, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    s = score[order]

    # last index of each run of equal scores -> one operating point per distinct threshold
    ends = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[ends]
    n_called = ends + 1
    precision = tp / n_called
    recall = tp / n_pos

    d_recall = np.diff(np.r_[0.0, recall])
    return float((d_recall * precision).sum())


def auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve by the rank identity, ties at average rank; nan when one class is absent."""
    y_true = np.asarray(y_true).astype(bool)
    score = np.asarray(score, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_scores = score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
