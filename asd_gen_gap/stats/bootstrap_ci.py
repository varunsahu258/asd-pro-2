"""Subject-level bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def bootstrap_ci(
    predictions_df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    n_resamples: int = 5_000,
    *,
    random_state: int | np.random.Generator | None = None,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Return a percentile CI after resampling subjects with replacement.

    A subject can have multiple prediction rows; all of its rows are retained
    together in each resample, preserving within-subject dependence.
    """
    if "subject_id" not in predictions_df.columns:
        raise ValueError("predictions_df must contain subject_id")
    if predictions_df.empty:
        raise ValueError("predictions_df must not be empty")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    point_estimate = float(metric_fn(predictions_df))
    if not np.isfinite(point_estimate):
        raise ValueError("metric_fn must return a finite point estimate")
    subjects = predictions_df["subject_id"].drop_duplicates().to_numpy()
    groups = {subject: group for subject, group in predictions_df.groupby("subject_id", sort=False)}
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    estimates = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        sample = pd.concat([groups[subject] for subject in sampled_subjects], ignore_index=True)
        estimates[index] = float(metric_fn(sample))
    estimates = estimates[np.isfinite(estimates)]
    if not len(estimates):
        raise ValueError("metric_fn did not produce a finite bootstrap estimate")
    alpha = (1 - confidence_level) / 2
    ci_low, ci_high = np.quantile(estimates, [alpha, 1 - alpha])
    return point_estimate, float(ci_low), float(ci_high)
