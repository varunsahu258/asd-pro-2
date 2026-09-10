"""Internal-to-external generalization-gap reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from asd_gen_gap.stats.bootstrap_ci import bootstrap_ci


WIDE_EXTERNAL_CI_WIDTH = 0.20
REQUIRED_COLUMNS = {"subject_id", "site", "dx_group", "y_prob", "y_pred", "model_name"}


def _read(path: Path, model_name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if missing := REQUIRED_COLUMNS.difference(frame.columns):
        raise ValueError(f"{path} is missing prediction columns: {sorted(missing)}")
    if frame.empty or not frame["subject_id"].is_unique:
        raise ValueError(f"{path} must contain one prediction per subject")
    if set(frame["model_name"].dropna().unique()) != {model_name}:
        raise ValueError(f"{path} must contain only model_name={model_name!r}")
    return frame


def _auc(frame: pd.DataFrame) -> float:
    return float(roc_auc_score(frame["dx_group"], frame["y_prob"])) if frame["dx_group"].nunique() == 2 else float("nan")


def _metrics(frame: pd.DataFrame, n_resamples: int, random_state: int) -> dict[str, float]:
    accuracy, accuracy_low, accuracy_high = bootstrap_ci(
        frame, lambda sample: accuracy_score(sample["dx_group"], sample["y_pred"]), n_resamples, random_state=random_state
    )
    auc = _auc(frame)
    if np.isfinite(auc):
        _, auc_low, auc_high = bootstrap_ci(frame, _auc, n_resamples, random_state=random_state)
    else:
        auc_low = auc_high = float("nan")
    return {"accuracy": accuracy, "accuracy_ci_low": accuracy_low, "accuracy_ci_high": accuracy_high,
            "auc": auc, "auc_ci_low": auc_low, "auc_ci_high": auc_high}


def generate_gap_report(predictions_dir: str | Path = "results/predictions", *,
                        output_path: str | Path = "results/gap_report.csv", n_resamples: int = 5_000,
                        wide_external_ci_width: float = WIDE_EXTERNAL_CI_WIDTH,
                        random_state: int = 0) -> pd.DataFrame:
    """Write pooled internal/external metrics and gaps for every paired model.

    ``ci_width_flag`` is ``wide`` when either external accuracy or external AUC
    CI exceeds ``wide_external_ci_width``.  If no model has both prediction
    files, a one-row status report is written instead of an empty CSV.
    """
    if not 0 < wide_external_ci_width:
        raise ValueError("wide_external_ci_width must be positive")
    directory = Path(predictions_dir)
    models = sorted({path.name.removesuffix("_internal.parquet") for path in directory.glob("*_internal.parquet") if (directory / f"{path.name.removesuffix('_internal.parquet')}_external.parquet").is_file()}) if directory.is_dir() else []
    rows: list[dict[str, object]] = []
    for index, model_name in enumerate(models):
        internal = _read(directory / f"{model_name}_internal.parquet", model_name)
        external = _read(directory / f"{model_name}_external.parquet", model_name)
        internal_metrics = _metrics(internal, n_resamples, random_state + index * 2)
        external_metrics = _metrics(external, n_resamples, random_state + index * 2 + 1)
        accuracy_width = external_metrics["accuracy_ci_high"] - external_metrics["accuracy_ci_low"]
        auc_width = external_metrics["auc_ci_high"] - external_metrics["auc_ci_low"]
        site_counts = external.groupby("site")["subject_id"].nunique().sort_index().to_dict()
        rows.append({
            "external_validation_status": "performed", "model_name": model_name,
            **{f"internal_{key}": value for key, value in internal_metrics.items()},
            **{f"external_{key}": value for key, value in external_metrics.items()},
            "accuracy_gap": internal_metrics["accuracy"] - external_metrics["accuracy"],
            "auc_gap": internal_metrics["auc"] - external_metrics["auc"],
            "external_ci_width": max(accuracy_width, auc_width),
            "ci_width_flag": "wide" if max(accuracy_width, auc_width) > wide_external_ci_width else "narrow",
            "external_site_counts": json.dumps(site_counts),
        })
    report = pd.DataFrame(rows) if rows else pd.DataFrame([{"external_validation_status": "no external validation performed"}])
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(destination, index=False)
    return report
