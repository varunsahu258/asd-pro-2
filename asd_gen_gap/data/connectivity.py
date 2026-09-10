"""Functional-connectivity features derived from ROI time series."""

from __future__ import annotations

import logging

import numpy as np


LOGGER = logging.getLogger(__name__)


def compute_connectivity(timeseries: np.ndarray) -> np.ndarray:
    """Return a Fisher-z Pearson-connectivity matrix for ROI time series.

    ROI columns containing non-finite values or zero variance cannot yield a
    meaningful Pearson correlation.  Their incident edges are set to zero rather
    than allowing NaNs to propagate into downstream feature matrices.
    """

    values = np.asarray(timeseries, dtype=float)
    if values.ndim != 2:
        raise ValueError(
            "timeseries must be a two-dimensional (timepoints, n_rois) array; "
            f"received {values.ndim} dimensions."
        )
    n_timepoints, n_rois = values.shape
    if n_timepoints < 2:
        raise ValueError("timeseries must contain at least two timepoints.")

    finite = np.isfinite(values).all(axis=0)
    nonconstant = np.zeros(n_rois, dtype=bool)
    if finite.any():
        nonconstant[finite] = np.ptp(values[:, finite], axis=0) > 0
    valid = finite & nonconstant
    invalid_count = int((~valid).sum())
    if invalid_count:
        LOGGER.warning(
            "Setting connectivity edges to zero for %d ROI time series with "
            "NaN/non-finite or constant variance values.",
            invalid_count,
        )

    correlations = np.zeros((n_rois, n_rois), dtype=float)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size > 1:
        valid_correlations = np.corrcoef(values[:, valid_indices], rowvar=False)
        correlations[np.ix_(valid_indices, valid_indices)] = valid_correlations

    # Avoid infinite z-values for numerically exact +/-1 correlations.
    limit = np.nextafter(1.0, 0.0)
    fisher_z = np.arctanh(np.clip(correlations, -limit, limit))
    np.fill_diagonal(fisher_z, 0.0)
    return fisher_z


def flatten_upper_triangle(matrix: np.ndarray) -> dict[str, float]:
    """Flatten the strictly upper triangle using stable ``conn_<i>_<j>`` keys."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be a square two-dimensional array.")

    row_indices, column_indices = np.triu_indices(values.shape[0], k=1)
    return {
        f"conn_{row}_{column}": float(values[row, column])
        for row, column in zip(row_indices, column_indices, strict=True)
    }
