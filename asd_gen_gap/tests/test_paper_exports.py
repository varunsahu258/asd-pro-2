"""Tests for graceful paper-artifact exports before evaluations have run."""

from pathlib import Path

import pandas as pd

from asd_gen_gap.paper.export_figures import export_paper_figures
from asd_gen_gap.paper.export_tables import export_paper_tables


def test_paper_exports_run_without_evaluation_artifacts(tmp_path: Path) -> None:
    table = export_paper_tables(tmp_path)
    text = table.read_text(encoding="utf-8")
    assert "Internal bake-off results are not available." in text
    assert "No external validation performed" in text
    assert export_paper_figures(tmp_path) == []


def test_paper_tables_include_reviewer_relevant_new_gap_metrics(tmp_path: Path) -> None:
    pd.DataFrame([{
        "external_validation_status": "performed", "model_name": "hcan",
        "internal_auc": 0.8, "external_auc": 0.7, "auc_gap": 0.1,
        "accuracy_gap": 0.1, "sensitivity_gap": 0.2, "f1_gap": 0.15,
        "external_ci_width": 0.1, "ci_width_flag": "narrow", "external_site_counts": "{}",
    }]).to_csv(tmp_path / "gap_report.csv", index=False)

    table = export_paper_tables(tmp_path)
    text = table.read_text(encoding="utf-8")

    assert "sensitivity_gap" in text
    assert "f1_gap" in text


def test_paper_figures_export_external_confusion_matrices_and_calibration(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for model_name in ("baseline", "hcan"):
        pd.DataFrame({"dx_group": [0, 1, 0, 1], "y_pred": [0, 1, 1, 1], "y_prob": [0.1, 0.8, 0.6, 0.9]}).to_parquet(
            predictions / f"{model_name}_external.parquet", index=False
        )

    outputs = export_paper_figures(tmp_path)

    for filename in ("figure_external_confusion_matrices.png", "figure_calibration_external.png"):
        path = tmp_path / filename
        assert path in outputs
        assert path.is_file()
