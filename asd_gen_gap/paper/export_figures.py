"""Export concise manuscript figures from prediction and gap-report artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


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


def export_paper_figures(results_dir: str | Path = "results") -> list[Path]:
    """Write a per-site internal-AUC chart and an internal/external AUC chart.

    Returns no paths when the corresponding input artifact does not yet exist;
    this lets paper export run before evaluation without fabricating figures.
    """
    results = Path(results_dir)
    site_auc = _internal_site_auc(results)
    gap_path = results / "gap_report.csv"
    gap = pd.read_csv(gap_path) if gap_path.is_file() else pd.DataFrame()
    if site_auc.empty and gap.empty:
        return []
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    if not site_auc.empty:
        labels = [f"{row.site}\n{row.model_name}" for row in site_auc.itertuples()]
        figure, axis = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4))
        axis.bar(labels, site_auc["auc"], color="#4472C4")
        axis.set_ylim(0, 1); axis.set_ylabel("ROC AUC"); axis.set_title("Internal LOSO AUC by site")
        figure.tight_layout()
        output = results / "figure_internal_site_auc.png"; figure.savefig(output, dpi=300); plt.close(figure)
        outputs.append(output)
    if not gap.empty and "internal_auc" in gap and "external_auc" in gap:
        figure, axis = plt.subplots(figsize=(5, 4))
        positions = range(len(gap))
        axis.bar([position - 0.18 for position in positions], gap["internal_auc"], width=0.36, label="Internal")
        axis.bar([position + 0.18 for position in positions], gap["external_auc"], width=0.36, label="External")
        axis.set_xticks(list(positions), gap["model_name"]); axis.set_ylim(0, 1); axis.set_ylabel("ROC AUC")
        axis.set_title("Internal versus external AUC"); axis.legend(); figure.tight_layout()
        output = results / "figure_internal_external_auc.png"; figure.savefig(output, dpi=300); plt.close(figure)
        outputs.append(output)
    return outputs
