"""Load ABIDE I PCP phenotypes and locate CC200 time-series files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


_COLUMN_MAP = {
    "SUB_ID": "subject_id",
    "SITE_ID": "site",
    "DX_GROUP": "dx_group",
    "AGE_AT_SCAN": "age",
    "SEX": "sex",
    "FIQ": "fiq",
    "func_mean_fd": "func_mean_fd",
}


def _config_value(config: Any, *keys: str) -> Any:
    """Read a nested setting from a mapping or a Pydantic-style config object."""

    value = config
    for key in keys:
        value = value[key] if isinstance(value, dict) else getattr(value, key)
    return value


def _phenotype_csv(abide_root: Path) -> Path:
    candidates = sorted(
        path
        for path in abide_root.rglob("*.csv")
        if path.name.lower().startswith("phenotypic")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No phenotypic CSV was found under ABIDE I directory: {abide_root}"
        )
    return candidates[0]


def _timeseries_path(abide_root: Path, subject_id: str) -> Path | None:
    matches = sorted(abide_root.rglob(f"{subject_id}_rois_cc200.1D"))
    return matches[0] if matches else None


def load_abide_i(config: Any) -> pd.DataFrame:
    """Load and standardize ABIDE I records, attaching each CC200 file path.

    The function only locates files; callers can load a selected time series with
    :func:`numpy.loadtxt` when constructing connectivity features.
    """

    abide_root = Path(_config_value(config, "dataset_paths", "abide_i"))
    if not abide_root.is_dir():
        raise FileNotFoundError(f"ABIDE I directory does not exist: {abide_root}")

    phenotype_path = _phenotype_csv(abide_root)
    phenotypes = pd.read_csv(phenotype_path, dtype={"SUB_ID": "string"})
    missing_columns = [column for column in _COLUMN_MAP if column not in phenotypes.columns]
    if missing_columns:
        raise ValueError(
            f"Phenotypic CSV {phenotype_path} is missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    loaded = phenotypes.loc[:, list(_COLUMN_MAP)].rename(columns=_COLUMN_MAP).copy()
    loaded["subject_id"] = loaded["subject_id"].astype("string").str.strip()
    paths = loaded["subject_id"].map(
        lambda subject_id: _timeseries_path(abide_root, str(subject_id))
    )
    missing_subjects = loaded.loc[paths.isna(), "subject_id"].astype(str).tolist()
    if missing_subjects:
        preview = ", ".join(missing_subjects[:5])
        suffix = "" if len(missing_subjects) <= 5 else ", ..."
        raise FileNotFoundError(
            f"Missing CC200 .1D time-series files for {len(missing_subjects)} subject(s): "
            f"{preview}{suffix}"
        )

    loaded["timeseries_path"] = paths.map(str)
    return loaded
