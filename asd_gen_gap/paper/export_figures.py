"""Export concise manuscript figures from prediction and gap-report artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_auc_score


def _internal_site_auc(results: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    predictions = results / "predictions"
    if not predictions.is_dir():
        return pd.DataFrame(rows)
    for path in sorted(predictions.glob("*_internal.parquet")):
        model = path.name.removesuffix("_internal.parquet")
        frame = pd.read_parquet(path)
        for site, group in frame.groupby("site"):
            if group["dx_group"].nunique() == 2:
                rows.append({"model_name": model, "site": site, "auc": roc_auc_score(group["dx_group"], group["y_prob"])})
    return pd.DataFrame(rows)


def _external_model_predictions(results: Path) -> dict[str, pd.DataFrame]:
    """Discover external prediction artifacts keyed by their model names."""
    directory = results / "predictions"
    if not directory.is_dir():
        return {}
    return {
        path.name.removesuffix("_external.parquet"): pd.read_parquet(path)
        for path in sorted(directory.glob("*_external.parquet"))
    }


def _external_confusion_matrices(results: Path) -> dict[str, np.ndarray]:
    """Compute binary external confusion matrices for every available model."""
    matrices: dict[str, np.ndarray] = {}
    for model_name, frame in _external_model_predictions(results).items():
        valid = frame.loc[frame["dx_group"].isin([0, 1])]
        if not valid.empty:
            matrices[model_name] = confusion_matrix(valid["dx_group"], valid["y_pred"], labels=[0, 1])
    return matrices


def export_paper_figures(results_dir: str | Path = "results") -> list[Path]:
    """Write a per-site internal-AUC chart and an internal/external AUC chart.

    Returns no paths when the corresponding input artifact does not yet exist;
    this lets paper export run before evaluation without fabricating figures.
    """
    results = Path(results_dir)
    site_auc = _internal_site_auc(results)
    external_predictions = _external_model_predictions(results)
    gap_path = results / "gap_report.csv"
    gap = pd.read_csv(gap_path) if gap_path.is_file() else pd.DataFrame()
    if site_auc.empty and gap.empty and not external_predictions:
        return []
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    if not site_auc.empty:
        models = sorted(site_auc["model_name"].unique())
        sites = sorted(site_auc["site"].unique())
        n_models = len(models)
        width = 0.8 / n_models
        positions = list(range(len(sites)))
        figure, axis = plt.subplots(figsize=(max(6, len(sites) * 0.8), 4))
        for index, model_name in enumerate(models):
            values = site_auc.loc[site_auc["model_name"] == model_name].set_index("site")["auc"].reindex(sites)
            offsets = [position - 0.4 + width / 2 + index * width for position in positions]
            axis.bar(offsets, values, width=width, color="#4472C4", label=model_name)
        axis.set_xticks(positions, sites)
        axis.set_ylim(0, 1); axis.set_ylabel("ROC AUC"); axis.set_title("Internal LOSO AUC by site")
        axis.legend()
        figure.tight_layout()
        output = results / "figure_internal_site_auc.png"; figure.savefig(output, dpi=300); plt.close(figure)
        outputs.append(output)
    if not gap.empty and "internal_auc" in gap and "external_auc" in gap:
        figure, axis = plt.subplots(figsize=(5, 4))
        models = sorted(gap["model_name"].unique())
        n_models = len(models)
        width = 0.8 / n_models
        positions = [0, 1]
        for index, model_name in enumerate(models):
            row = gap.loc[gap["model_name"] == model_name].iloc[0]
            offsets = [position - 0.4 + width / 2 + index * width for position in positions]
            axis.bar(offsets, [row["internal_auc"], row["external_auc"]], width=width, label=model_name)
        axis.set_xticks(positions, ["Internal", "External"]); axis.set_ylim(0, 1); axis.set_ylabel("ROC AUC")
        axis.set_title("Internal versus external AUC"); axis.legend(); figure.tight_layout()
        output = results / "figure_internal_external_auc.png"; figure.savefig(output, dpi=300); plt.close(figure)
        outputs.append(output)
    matrices = _external_confusion_matrices(results)
    if matrices:
        figure, axes = plt.subplots(1, len(matrices), figsize=(4 * len(matrices), 4))
        axes = [axes] if len(matrices) == 1 else list(axes)
        for axis, (model_name, matrix) in zip(axes, matrices.items(), strict=True):
            axis.imshow(matrix, cmap="Blues")
            for row, column in np.ndindex(matrix.shape):
                axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
            axis.set_xticks([0, 1], [0, 1]); axis.set_yticks([0, 1], [0, 1])
            axis.set_xlabel("Predicted"); axis.set_ylabel("True"); axis.set_title(model_name)
        figure.tight_layout()
        output = results / "figure_external_confusion_matrices.png"; figure.savefig(output, dpi=300); plt.close(figure)
        outputs.append(output)
    if external_predictions:
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
        plotted = False
        for model_name, frame in external_predictions.items():
            valid = frame.loc[frame["dx_group"].isin([0, 1]) & frame["y_prob"].notna()]
            if valid.empty:
                continue
            fraction_positive, mean_predicted = calibration_curve(valid["dx_group"], valid["y_prob"], n_bins=10)
            axis.plot(mean_predicted, fraction_positive, marker="o", label=model_name)
            plotted = True
        if plotted:
            axis.set_xlabel("Mean predicted probability"); axis.set_ylabel("Fraction of positives")
            axis.set_title("External calibration"); axis.legend(); figure.tight_layout()
            output = results / "figure_calibration_external.png"; figure.savefig(output, dpi=300); outputs.append(output)
        plt.close(figure)
    return outputs
