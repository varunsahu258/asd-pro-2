"""Tests for internal-to-external performance-gap and limitations reporting."""

from pathlib import Path

import pandas as pd

from asd_gen_gap.eval.gap_report import generate_gap_report
from asd_gen_gap.paper.limitations import generate_limitations


REAL_SITE_COUNTS = {"ABIDEII-UCLA_1": 32, "ABIDEII-KUL_3": 28, "ABIDEII-U_MIA_1": 28, "ABIDEII-NYU_2": 27}


def _predictions(model: str, site_counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    for site, count in site_counts.items():
        for index in range(count):
            label = index % 2
            rows.append({"subject_id": f"{site}-{index}", "site": site, "dx_group": label,
                         "y_prob": 0.8 if label else 0.2, "y_pred": label, "model_name": model})
    return pd.DataFrame(rows)


def test_gap_report_writes_explicit_no_external_status(tmp_path: Path) -> None:
    _predictions("baseline", {"internal": 4}).to_parquet(tmp_path / "baseline_internal.parquet", index=False)
    report = generate_gap_report(tmp_path, output_path=tmp_path / "gap_report.csv", n_resamples=20)
    assert report.loc[0, "external_validation_status"] == "no external validation performed"
    assert "No external validation was performed" in generate_limitations(tmp_path)


def test_combined_limitations_uses_real_site_counts_wide_ci_and_pipeline_note(tmp_path: Path) -> None:
    internal = _predictions("baseline", {"internal": 40})
    external = _predictions("baseline", REAL_SITE_COUNTS)
    external["y_prob"] = [0.9 if index % 3 == 0 else 0.1 for index in range(len(external))]
    external["y_pred"] = (external["y_prob"] >= 0.5).astype(int)
    internal.to_parquet(tmp_path / "baseline_internal.parquet", index=False)
    external.to_parquet(tmp_path / "baseline_external.parquet", index=False)
    report = generate_gap_report(tmp_path, output_path=tmp_path / "gap_report.csv", n_resamples=50, wide_external_ci_width=0.01)
    (tmp_path / "abide_ii_preproc_manifest.csv").write_text("subject_id,status\na,ok\n", encoding="utf-8")

    text = generate_limitations(tmp_path)
    metric_columns = {
        "internal_sensitivity", "external_sensitivity", "sensitivity_gap",
        "internal_specificity", "external_specificity", "specificity_gap",
        "internal_f1", "external_f1", "f1_gap",
        "internal_balanced_accuracy", "external_balanced_accuracy", "balanced_accuracy_gap",
        "internal_pr_auc", "external_pr_auc", "pr_auc_gap",
        "internal_brier", "external_brier", "brier_gap",
    }
    assert metric_columns.issubset(report.columns)
    metric_values = report.loc[:, [
        column for column in report.columns
        if any(metric in column for metric in ("sensitivity", "specificity", "f1", "balanced_accuracy"))
        and not column.endswith("_gap")
    ]]
    assert metric_values.apply(lambda column: column.between(0, 1) | column.isna()).all().all()
    assert report.loc[0, "ci_width_flag"] == "wide"
    assert "imprecise" in text
    assert "custom minimal pipeline" in text
    assert "ABIDEII-KUL_3 (n=28)" in text
    assert "ABIDEII-U_MIA_1 (n=28)" in text
    assert "ABIDEII-NYU_2 (n=27)" in text
    assert "ABIDEII-UCLA_1" not in text
    assert "summary-level mean framewise-displacement exclusion" in text
