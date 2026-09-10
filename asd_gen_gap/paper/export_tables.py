"""Export manuscript-ready tables from evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def export_paper_tables(results_dir: str | Path = "results", *, output_path: str | Path | None = None) -> Path:
    """Combine the internal bake-off and external gap report into Markdown.

    The internal bake-off artifact already includes its per-site rows, so it is
    embedded verbatim rather than re-computed with potentially different
    rounding. Missing evaluation artifacts are represented explicitly.
    """
    results = Path(results_dir)
    destination = Path(output_path) if output_path is not None else results / "paper_tables.md"
    sections = ["# Paper tables"]
    internal = results / "table_internal_bakeoff.md"
    if internal.is_file():
        sections.extend(["## Table 1. Internal LOSO bake-off and per-site breakdown", internal.read_text(encoding="utf-8").strip()])
    else:
        sections.extend(["## Table 1. Internal LOSO bake-off and per-site breakdown", "Internal bake-off results are not available."])
    gap = results / "gap_report.csv"
    sections.append("## Table 2. External validation and internal-to-external gap")
    if not gap.is_file():
        sections.append("No external validation performed (gap report is not available).")
    else:
        report = pd.read_csv(gap)
        if "external_validation_status" in report and (report["external_validation_status"] == "no external validation performed").any():
            sections.append("No external validation performed.")
        else:
            columns = [column for column in ("model_name", "internal_auc", "external_auc", "auc_gap", "pr_auc_gap", "accuracy_gap", "sensitivity_gap", "f1_gap", "external_ci_width", "ci_width_flag", "external_site_counts") if column in report]
            sections.append(_markdown(report.loc[:, columns]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return destination
