"""Create the internal baseline-versus-HCAN bake-off table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from asd_gen_gap.stats.bootstrap_ci import bootstrap_ci
from asd_gen_gap.stats.pairwise_tests import delong_test, holm_bonferroni, mcnemar_test


MODELS = ("baseline", "hcan")
REQUIRED_COLUMNS = {"subject_id", "site", "dx_group", "y_prob", "y_pred", "model_name"}


def _load_predictions(predictions_dir: str | Path) -> dict[str, pd.DataFrame]:
    directory = Path(predictions_dir)
    loaded: dict[str, pd.DataFrame] = {}
    for model_name in MODELS:
        path = directory / f"{model_name}_internal.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing internal predictions: {path}")
        frame = pd.read_parquet(path)
        if missing := REQUIRED_COLUMNS.difference(frame.columns):
            raise ValueError(f"{path} is missing prediction columns: {sorted(missing)}")
        if not frame["subject_id"].is_unique:
            raise ValueError(f"{path} must have one prediction per subject")
        if set(frame["model_name"].dropna().unique()) != {model_name}:
            raise ValueError(f"{path} must have model_name={model_name!r}")
        loaded[model_name] = frame
    return loaded


def _auc(frame: pd.DataFrame) -> float:
    """Return AUC, or NaN for a bootstrap draw containing one label class."""
    return float(roc_auc_score(frame["dx_group"], frame["y_prob"])) if frame["dx_group"].nunique() == 2 else float("nan")


def _performance_row(frame: pd.DataFrame, model_name: str, scope: str, site: object | None,
                     n_resamples: int, random_state: int) -> dict[str, object]:
    accuracy, accuracy_low, accuracy_high = bootstrap_ci(
        frame, lambda sample: accuracy_score(sample["dx_group"], sample["y_pred"]), n_resamples, random_state=random_state
    )
    auc = _auc(frame)
    if np.isfinite(auc):
        _, auc_low, auc_high = bootstrap_ci(frame, _auc, n_resamples, random_state=random_state)
    else:
        auc_low = auc_high = float("nan")
    return {
        "scope": scope, "site": "pooled" if site is None else site, "model_name": model_name,
        "n_subjects": len(frame), "accuracy": accuracy, "accuracy_ci_low": accuracy_low,
        "accuracy_ci_high": accuracy_high, "auc": auc, "auc_ci_low": auc_low, "auc_ci_high": auc_high,
    }


def _paired_tests(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    baseline, hcan = predictions["baseline"], predictions["hcan"]
    paired = baseline.merge(hcan, on="subject_id", suffixes=("_baseline", "_hcan"), validate="one_to_one")
    if len(paired) != len(baseline) or len(paired) != len(hcan):
        raise ValueError("Baseline and HCAN predictions must cover the same subjects")
    if not (paired["dx_group_baseline"] == paired["dx_group_hcan"]).all():
        raise ValueError("Paired predictions disagree on dx_group")
    labels = paired["dx_group_baseline"].to_numpy()
    values = np.sort(np.unique(labels))
    if len(values) != 2:
        raise ValueError("Paired comparisons require exactly two dx_group classes")
    binary_labels = (labels == values[1]).astype(int)
    _, _, delong_p = delong_test(binary_labels, paired["y_prob_baseline"], paired["y_prob_hcan"])
    _, mcnemar_p = mcnemar_test(labels, paired["y_pred_baseline"], paired["y_pred_hcan"])
    adjusted, rejected = holm_bonferroni([delong_p, mcnemar_p])
    return pd.DataFrame({
        "comparison": ["baseline vs hcan", "baseline vs hcan"],
        "test": ["DeLong AUC", "McNemar accuracy"],
        "p_value": [delong_p, mcnemar_p],
        "p_value_holm": adjusted,
        "reject_holm_0_05": rejected,
    })


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
    sites = sorted(set(predictions["baseline"]["site"]) | set(predictions["hcan"]["site"]))
    for site in sites:
        for index, (model_name, frame) in enumerate(predictions.items()):
            subset = frame.loc[frame["site"] == site]
            if subset.empty:
                raise ValueError(f"{model_name} has no predictions for site {site!r}")
            rows.append(_performance_row(subset, model_name, "site", site, n_resamples, random_state + index))
    table = pd.DataFrame(rows)
    table.attrs["pairwise_tests"] = _paired_tests(predictions)
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
                         n_resamples: int = 5_000, random_state: int = 0) -> pd.DataFrame:
    """Write pooled/per-site metrics and paired tests to the internal Markdown table."""
    table = build_bakeoff_table(predictions_dir, n_resamples=n_resamples, random_state=random_state)
    pairwise = table.attrs["pairwise_tests"]
    markdown = "# Internal model bake-off\n\n## Performance\n\n" + _markdown(table) + "\n\n## Paired comparisons\n\n" + _markdown(pairwise) + "\n"
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown, encoding="utf-8")
    return table
