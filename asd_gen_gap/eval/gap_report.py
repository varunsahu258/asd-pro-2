"""Internal-to-external generalization-gap reporting."""

from __future__ import annotations

from collections.abc import Callable
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, recall_score, roc_auc_score

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


def _sensitivity(frame: pd.DataFrame) -> float:
    return float(recall_score(frame["dx_group"], frame["y_pred"], pos_label=1, zero_division=0)) if frame["dx_group"].nunique() == 2 else float("nan")


def _specificity(frame: pd.DataFrame) -> float:
    return float(recall_score(frame["dx_group"], frame["y_pred"], pos_label=0, zero_division=0)) if frame["dx_group"].nunique() == 2 else float("nan")


def _f1(frame: pd.DataFrame) -> float:
    return float(f1_score(frame["dx_group"], frame["y_pred"], zero_division=0)) if frame["dx_group"].nunique() == 2 else float("nan")


def _balanced_accuracy(frame: pd.DataFrame) -> float:
    return float(balanced_accuracy_score(frame["dx_group"], frame["y_pred"])) if frame["dx_group"].nunique() == 2 else float("nan")


def _pr_auc(frame: pd.DataFrame) -> float:
    return float(average_precision_score(frame["dx_group"], frame["y_prob"])) if frame["dx_group"].nunique() == 2 else float("nan")


def _brier(frame: pd.DataFrame) -> float:
    return float(brier_score_loss(frame["dx_group"], frame["y_prob"]))


def _metric_with_ci(frame: pd.DataFrame, metric: Callable[[pd.DataFrame], float], n_resamples: int,
                    random_state: int) -> tuple[float, float, float]:
    point_estimate = metric(frame)
    if not np.isfinite(point_estimate):
        return point_estimate, float("nan"), float("nan")
    _, ci_low, ci_high = bootstrap_ci(frame, lambda sample: metric(sample), n_resamples, random_state=random_state)
    return point_estimate, ci_low, ci_high


def _metrics(frame: pd.DataFrame, n_resamples: int, random_state: int) -> dict[str, float]:
    accuracy, accuracy_low, accuracy_high = bootstrap_ci(
        frame, lambda sample: accuracy_score(sample["dx_group"], sample["y_pred"]), n_resamples, random_state=random_state
    )
    auc, auc_low, auc_high = _metric_with_ci(frame, _auc, n_resamples, random_state)
    sensitivity, sensitivity_low, sensitivity_high = _metric_with_ci(frame, _sensitivity, n_resamples, random_state)
    specificity, specificity_low, specificity_high = _metric_with_ci(frame, _specificity, n_resamples, random_state)
    f1, f1_low, f1_high = _metric_with_ci(frame, _f1, n_resamples, random_state)
    balanced_accuracy, balanced_accuracy_low, balanced_accuracy_high = _metric_with_ci(
        frame, _balanced_accuracy, n_resamples, random_state
    )
    pr_auc, pr_auc_low, pr_auc_high = _metric_with_ci(frame, _pr_auc, n_resamples, random_state)
    brier, brier_low, brier_high = _metric_with_ci(frame, _brier, n_resamples, random_state)
    return {"accuracy": accuracy, "accuracy_ci_low": accuracy_low, "accuracy_ci_high": accuracy_high,
            "auc": auc, "auc_ci_low": auc_low, "auc_ci_high": auc_high,
            "sensitivity": sensitivity, "sensitivity_ci_low": sensitivity_low, "sensitivity_ci_high": sensitivity_high,
            "specificity": specificity, "specificity_ci_low": specificity_low, "specificity_ci_high": specificity_high,
            "f1": f1, "f1_ci_low": f1_low, "f1_ci_high": f1_high,
            "balanced_accuracy": balanced_accuracy, "balanced_accuracy_ci_low": balanced_accuracy_low,
            "balanced_accuracy_ci_high": balanced_accuracy_high,
            "pr_auc": pr_auc, "pr_auc_ci_low": pr_auc_low, "pr_auc_ci_high": pr_auc_high,
            "brier": brier, "brier_ci_low": brier_low, "brier_ci_high": brier_high}


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
            "sensitivity_gap": internal_metrics["sensitivity"] - external_metrics["sensitivity"],
            "specificity_gap": internal_metrics["specificity"] - external_metrics["specificity"],
            "f1_gap": internal_metrics["f1"] - external_metrics["f1"],
            "balanced_accuracy_gap": internal_metrics["balanced_accuracy"] - external_metrics["balanced_accuracy"],
            "pr_auc_gap": internal_metrics["pr_auc"] - external_metrics["pr_auc"],
            "brier_gap": internal_metrics["brier"] - external_metrics["brier"],
            "external_ci_width": max(accuracy_width, auc_width),
            "ci_width_flag": "wide" if max(accuracy_width, auc_width) > wide_external_ci_width else "narrow",
            "external_site_counts": json.dumps(site_counts),
        })
    report = pd.DataFrame(rows) if rows else pd.DataFrame([{"external_validation_status": "no external validation performed"}])
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(destination, index=False)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", default="results/predictions")
    parser.add_argument("--out", default="results/gap_report.csv")
    parser.add_argument("--n-resamples", type=int, default=5_000)
    parser.add_argument("--wide-external-ci-width", type=float, default=WIDE_EXTERNAL_CI_WIDTH)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args(argv)
    generate_gap_report(args.predictions_dir, output_path=args.out, n_resamples=args.n_resamples,
                        wide_external_ci_width=args.wide_external_ci_width, random_state=args.random_state)


if __name__ == "__main__":
    main()
