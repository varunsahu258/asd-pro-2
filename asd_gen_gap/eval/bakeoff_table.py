"""Create the internal baseline-versus-HCAN bake-off table."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import argparse
from glob import glob

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, recall_score, roc_auc_score

from asd_gen_gap.stats.bootstrap_ci import bootstrap_ci
from asd_gen_gap.stats.pairwise_tests import delong_test, holm_bonferroni, mcnemar_test


REQUIRED_COLUMNS = {"subject_id", "site", "dx_group", "y_prob", "y_pred", "model_name"}


def _load_predictions(predictions_dir: str | Path) -> dict[str, pd.DataFrame]:
    directory = Path(predictions_dir)
    loaded: dict[str, pd.DataFrame] = {}
    for path in sorted(directory.glob("*_internal.parquet")):
        model_name = path.name.removesuffix("_internal.parquet")
        frame = pd.read_parquet(path)
        if missing := REQUIRED_COLUMNS.difference(frame.columns):
            raise ValueError(f"{path} is missing prediction columns: {sorted(missing)}")
        if not frame["subject_id"].is_unique:
            raise ValueError(f"{path} must have one prediction per subject")
        if set(frame["model_name"].dropna().unique()) != {model_name}:
            raise ValueError(f"{path} must have model_name={model_name!r}")
        loaded[model_name] = frame
    if "baseline" not in loaded:
        raise ValueError("Missing baseline internal predictions; baseline is required as the comparison anchor")
    if len(loaded) < 2:
        raise ValueError("At least two internal prediction models are required for a bake-off")
    return loaded


def _auc(frame: pd.DataFrame) -> float:
    """Return AUC, or NaN for a bootstrap draw containing one label class."""
    return float(roc_auc_score(frame["dx_group"], frame["y_prob"])) if frame["dx_group"].nunique() == 2 else float("nan")


def _sensitivity(frame: pd.DataFrame) -> float:
    """Return sensitivity, or NaN when only one true class is present."""
    return float(recall_score(frame["dx_group"], frame["y_pred"], pos_label=1, zero_division=0)) if frame["dx_group"].nunique() == 2 else float("nan")


def _specificity(frame: pd.DataFrame) -> float:
    """Return specificity, or NaN when only one true class is present."""
    return float(recall_score(frame["dx_group"], frame["y_pred"], pos_label=0, zero_division=0)) if frame["dx_group"].nunique() == 2 else float("nan")


def _f1(frame: pd.DataFrame) -> float:
    """Return F1, or NaN when only one true class is present."""
    return float(f1_score(frame["dx_group"], frame["y_pred"], zero_division=0)) if frame["dx_group"].nunique() == 2 else float("nan")


def _balanced_accuracy(frame: pd.DataFrame) -> float:
    """Return balanced accuracy, or NaN when only one true class is present."""
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


def _performance_row(frame: pd.DataFrame, model_name: str, scope: str, site: object | None,
                     n_resamples: int, random_state: int) -> dict[str, object]:
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
    return {
        "scope": scope, "site": "pooled" if site is None else site, "model_name": model_name,
        "n_subjects": len(frame), "accuracy": accuracy, "accuracy_ci_low": accuracy_low,
        "accuracy_ci_high": accuracy_high, "auc": auc, "auc_ci_low": auc_low, "auc_ci_high": auc_high,
        "sensitivity": sensitivity, "sensitivity_ci_low": sensitivity_low, "sensitivity_ci_high": sensitivity_high,
        "specificity": specificity, "specificity_ci_low": specificity_low, "specificity_ci_high": specificity_high,
        "f1": f1, "f1_ci_low": f1_low, "f1_ci_high": f1_high,
        "balanced_accuracy": balanced_accuracy, "balanced_accuracy_ci_low": balanced_accuracy_low,
        "balanced_accuracy_ci_high": balanced_accuracy_high,
        "pr_auc": pr_auc, "pr_auc_ci_low": pr_auc_low, "pr_auc_ci_high": pr_auc_high,
        "brier": brier, "brier_ci_low": brier_low, "brier_ci_high": brier_high,
    }


def _paired_tests(predictions: dict[str, pd.DataFrame], n_resamples: int, random_state: int) -> pd.DataFrame:
    baseline = predictions["baseline"]
    rows: list[dict[str, object]] = []
    p_values: list[float] = []
    effect_rows: list[dict[str, object]] = []
    for model_name, model_predictions in predictions.items():
        if model_name == "baseline":
            continue
        paired = baseline.merge(model_predictions, on="subject_id", suffixes=("_baseline", f"_{model_name}"), validate="one_to_one")
        if len(paired) != len(baseline) or len(paired) != len(model_predictions):
            raise ValueError(f"Baseline and {model_name} predictions must cover the same subjects")
        if not (paired["dx_group_baseline"] == paired[f"dx_group_{model_name}"]).all():
            raise ValueError("Paired predictions disagree on dx_group")
        labels = paired["dx_group_baseline"].to_numpy()
        values = np.sort(np.unique(labels))
        if len(values) != 2:
            raise ValueError("Paired comparisons require exactly two dx_group classes")
        binary_labels = (labels == values[1]).astype(int)
        _, _, delong_p = delong_test(binary_labels, paired["y_prob_baseline"], paired[f"y_prob_{model_name}"])
        _, mcnemar_p = mcnemar_test(labels, paired["y_pred_baseline"], paired[f"y_pred_{model_name}"])
        comparison = f"baseline vs {model_name}"
        rows.extend((
            {"comparison": comparison, "test": "DeLong AUC", "p_value": delong_p},
            {"comparison": comparison, "test": "McNemar accuracy", "p_value": mcnemar_p},
        ))
        p_values.extend((delong_p, mcnemar_p))
        rng = np.random.default_rng(random_state)
        deltas = []
        baseline_probabilities = paired["y_prob_baseline"].to_numpy()
        model_probabilities = paired[f"y_prob_{model_name}"].to_numpy()
        for _ in range(n_resamples):
            indices = rng.integers(0, len(paired), len(paired))
            sampled_labels = binary_labels[indices]
            if np.unique(sampled_labels).size == 2:
                deltas.append(roc_auc_score(sampled_labels, model_probabilities[indices]) - roc_auc_score(sampled_labels, baseline_probabilities[indices]))
        auc_delta = roc_auc_score(binary_labels, model_probabilities) - roc_auc_score(binary_labels, baseline_probabilities)
        ci_low, ci_high = np.quantile(deltas, [0.025, 0.975]) if deltas else (float("nan"), float("nan"))
        effect_rows.append({"comparison": comparison, "test": "AUC delta (paired bootstrap)", "auc_delta": auc_delta,
                            "auc_delta_ci_low": ci_low, "auc_delta_ci_high": ci_high})
    adjusted, rejected = holm_bonferroni(p_values)
    for row, adjusted_value, rejected_value in zip(rows, adjusted, rejected, strict=True):
        row["p_value_holm"] = adjusted_value
        row["reject_holm_0_05"] = rejected_value
    rows.extend(effect_rows)
    return pd.DataFrame(rows, columns=["comparison", "test", "p_value", "p_value_holm", "reject_holm_0_05", "auc_delta", "auc_delta_ci_low", "auc_delta_ci_high"])


def build_bakeoff_table(predictions_dir: str | Path = "results/predictions", *, n_resamples: int = 5_000,
                        random_state: int = 0) -> pd.DataFrame:
    """Compute pooled and per-site internal performance with bootstrap CIs.

    The returned dataframe has pooled and per-site rows. Paired DeLong and
    McNemar results (with Holm--Bonferroni adjustment) are attached in
    ``table.attrs['pairwise_tests']`` for use by :func:`write_bakeoff_table`.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    predictions = _load_predictions(predictions_dir)
    rows = [
        _performance_row(frame, model_name, "pooled", None, n_resamples, random_state + index)
        for index, (model_name, frame) in enumerate(predictions.items())
    ]
    sites = sorted(set().union(*(set(frame["site"]) for frame in predictions.values())))
    for site in sites:
        for index, (model_name, frame) in enumerate(predictions.items()):
            subset = frame.loc[frame["site"] == site]
            if subset.empty:
                raise ValueError(f"{model_name} has no predictions for site {site!r}")
            rows.append(_performance_row(subset, model_name, "site", site, n_resamples, random_state + index))
    table = pd.DataFrame(rows)
    table.attrs["pairwise_tests"] = _paired_tests(predictions, n_resamples, random_state)
    return table


def _markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    formatted = frame.copy()
    for column in formatted.select_dtypes(include="number"):
        formatted[column] = formatted[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "NA")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in formatted.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_bakeoff_table(predictions_dir: str | Path = "results/predictions", *,
                         output_path: str | Path = "results/table_internal_bakeoff.md",
                         n_resamples: int = 5_000, random_state: int = 0,
                         multiseed_glob: str | None = None) -> pd.DataFrame:
    """Write pooled/per-site metrics and paired tests to the internal Markdown table."""
    table = build_bakeoff_table(predictions_dir, n_resamples=n_resamples, random_state=random_state)
    pairwise = table.attrs["pairwise_tests"]
    markdown = "# Internal model bake-off\n\n## Performance\n\n" + _markdown(table) + "\n\n## Paired comparisons\n\n" + _markdown(pairwise) + "\n"
    if multiseed_glob:
        paths = [Path(path) for path in sorted(glob(multiseed_glob))]
        if len(paths) >= 2:
            from asd_gen_gap.eval.multiseed import aggregate_seed_runs

            stability = aggregate_seed_runs(paths, model_name="hcan")
            markdown += "\n## Seed stability (HCAN)\n\n" + _markdown(stability) + "\n"
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown, encoding="utf-8")
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", default="results/predictions")
    parser.add_argument("--out", default="results/table_internal_bakeoff.md")
    parser.add_argument("--n-resamples", type=int, default=5_000)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--multiseed-glob", default=None)
    args = parser.parse_args(argv)
    write_bakeoff_table(args.predictions_dir, output_path=args.out, n_resamples=args.n_resamples,
                        random_state=args.random_state, multiseed_glob=args.multiseed_glob)


if __name__ == "__main__":
    main()
