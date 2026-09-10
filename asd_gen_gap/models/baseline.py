"""Sklearn baseline models used in leave-one-site-out experiments."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLINICAL_FEATURES = ("age_site_z", "sex", "fiq_site_z")


def baseline_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return connectivity and clinical columns for the logistic baseline.

    The input dataframe is deliberately used only to discover column names;
    scaling and fitting are performed later, inside the training-fold pipeline.
    """
    connection_columns = sorted(column for column in frame.columns if column.startswith("conn_"))
    missing = [column for column in CLINICAL_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required baseline feature columns: {missing}")
    if not connection_columns:
        raise ValueError("At least one conn_<i>_<j> feature is required")
    return [*connection_columns, *CLINICAL_FEATURES]


def make_baseline_model(*, random_state: int | None = 0, C: float = 1.0) -> Pipeline:
    """Create the standardized logistic-regression baseline.

    ``Pipeline`` ensures that :class:`StandardScaler` is fit by ``fit`` on a
    LOSO training partition only, rather than leaking held-out-site statistics.
    """
    model = Pipeline(
        steps=[
            ("standardize", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(C=C, max_iter=1_000, random_state=random_state),
            ),
        ]
    )
    model.model_name = "logistic_regression_baseline"
    return model


def baseline_model() -> Pipeline:
    """Compatibility-friendly zero-argument model factory for ``run_loso``."""
    return make_baseline_model()
