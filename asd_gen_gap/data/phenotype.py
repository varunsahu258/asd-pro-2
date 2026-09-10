"""Phenotype normalization and subject-level motion exclusion."""

from __future__ import annotations

import pandas as pd


def normalize_phenotypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce clinical values and add age/FIQ z-scores within each site."""

    normalized = df.copy()
    normalized["dx_group"] = pd.to_numeric(normalized["dx_group"], errors="raise").astype(int)
    for column in ("age", "fiq"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        site_mean = normalized.groupby("site")[column].transform("mean")
        site_std = normalized.groupby("site")[column].transform("std")
        normalized[f"{column}_site_z"] = (normalized[column] - site_mean) / site_std
    return normalized


def filter_by_motion(
    df: pd.DataFrame, threshold: float = 0.2
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Exclude subjects whose mean framewise displacement exceeds ``threshold``.

    This is summary-level subject exclusion, not frame-level scrubbing: the
    standard PCP download supplies summary ``func_mean_fd``, not per-frame FD.
    """

    if "func_mean_fd" not in df:
        raise ValueError("Cannot filter by motion: 'func_mean_fd' column is required.")
    mean_fd = pd.to_numeric(df["func_mean_fd"], errors="coerce")
    excluded_mask = mean_fd > threshold
    excluded_ids = df.loc[excluded_mask, "subject_id"].astype(str).tolist()
    filtered = df.loc[~excluded_mask].copy()
    report: dict[str, object] = {
        "threshold": threshold,
        "excluded_count": len(excluded_ids),
        "included_count": len(filtered),
        "excluded_subject_ids": excluded_ids,
        "method": "summary-level subject exclusion (not frame-level scrubbing)",
    }
    return filtered, report
