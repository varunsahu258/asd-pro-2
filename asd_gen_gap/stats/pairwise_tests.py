"""Paired statistical tests for comparing model predictions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import chi2, norm


def _midranks(values: np.ndarray) -> np.ndarray:
    """Compute one-indexed midranks, assigning equal values the same rank."""
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    return ranks


def _delong_covariance(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return AUCs and their DeLong covariance matrix for classifiers in rows."""
    positive = y_true == 1
    m, n = int(positive.sum()), int((~positive).sum())
    if m < 2 or n < 2:
        raise ValueError("DeLong's test requires at least two positive and two negative labels")
    ordered = np.concatenate((scores[:, positive], scores[:, ~positive]), axis=1)
    tx = np.vstack([_midranks(row[:m]) for row in ordered])
    ty = np.vstack([_midranks(row[m:]) for row in ordered])
    tz = np.vstack([_midranks(row) for row in ordered])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    # np.cov returns a scalar for a single classifier; force a 2-D result.
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(
    y_true: Sequence[int], y_score_a: Sequence[float], y_score_b: Sequence[float]
) -> tuple[float, float, float]:
    """Compare two correlated ROC AUCs with DeLong's two-sided test.

    Returns ``(auc_a, auc_b, p_value)``. Scores must be paired predictions for
    the same binary-labelled subjects, where 1 is the positive class.
    """
    labels = np.asarray(y_true)
    scores = np.asarray([y_score_a, y_score_b], dtype=float)
    if labels.ndim != 1 or scores.ndim != 2 or scores.shape[1] != len(labels):
        raise ValueError("labels and both score arrays must be one-dimensional and equally sized")
    if len(labels) < 2 or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("y_true must contain binary 0/1 labels")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    aucs, covariance = _delong_covariance(labels, scores)
    variance = float(np.array([1.0, -1.0]) @ covariance @ np.array([1.0, -1.0]))
    difference = float(aucs[0] - aucs[1])
    if variance <= 0:
        # Identical scores have a zero difference and no evidence against H0.
        p_value = 1.0 if np.isclose(difference, 0.0) else 0.0
    else:
        p_value = float(2 * norm.sf(abs(difference) / np.sqrt(variance)))
    return float(aucs[0]), float(aucs[1]), p_value


def mcnemar_test(
    y_true: Sequence[int], y_pred_a: Sequence[int], y_pred_b: Sequence[int], *, correction: bool = True
) -> tuple[float, float]:
    """Perform McNemar's chi-squared test for paired classifier accuracy.

    Returns ``(statistic, p_value)``; by default the usual continuity
    correction is used. The exact binomial version is preferable for very
    small discordant counts, but this implementation intentionally matches the
    common chi-squared textbook formulation.
    """
    truth, first, second = (np.asarray(values) for values in (y_true, y_pred_a, y_pred_b))
    if truth.ndim != 1 or first.shape != truth.shape or second.shape != truth.shape:
        raise ValueError("y_true and predictions must be equally sized one-dimensional arrays")
    first_correct, second_correct = first == truth, second == truth
    b = int(np.sum(first_correct & ~second_correct))
    c = int(np.sum(~first_correct & second_correct))
    if b + c == 0:
        return 0.0, 1.0
    numerator = abs(b - c) - 1 if correction else abs(b - c)
    statistic = numerator**2 / (b + c)
    return float(statistic), float(chi2.sf(statistic, df=1))


def holm_bonferroni(p_values: Sequence[float], *, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Apply Holm--Bonferroni correction and return adjusted p-values/rejects."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("p_values must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must lie between zero and one")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    order = np.argsort(values)
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate((len(values) - np.arange(len(values))) * sorted_values)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted, adjusted <= alpha
