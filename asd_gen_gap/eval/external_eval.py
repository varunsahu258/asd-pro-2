"""Summarize one-shot external baseline-versus-HCAN predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def evaluate_external_predictions(predictions_dir: str | Path = "results/predictions") -> pd.DataFrame:
    """Return external accuracy/AUC rows from baseline and HCAN files.

    Expected filenames are ``baseline_external.parquet`` and
    ``hcan_external.parquet``; the expected model names are consequently
    ``baseline`` and ``hcan``.
    """
    directory = Path(predictions_dir)
    rows: list[dict[str, object]] = []
    for model_name in ("baseline", "hcan"):
        path = directory / f"{model_name}_external.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing external predictions: {path}")
        predictions = pd.read_parquet(path)
        required = {"dx_group", "y_prob", "y_pred"}
        if missing := required.difference(predictions.columns):
            raise ValueError(f"{path} is missing prediction columns: {sorted(missing)}")
        rows.append({
            "model_name": model_name,
            "n_subjects": len(predictions),
            "accuracy": accuracy_score(predictions["dx_group"], predictions["y_pred"]),
            "auc": roc_auc_score(predictions["dx_group"], predictions["y_prob"]),
        })
    return pd.DataFrame(rows)
