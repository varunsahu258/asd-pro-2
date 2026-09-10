"""Tests for graceful paper-artifact exports before evaluations have run."""

from pathlib import Path

from asd_gen_gap.paper.export_figures import export_paper_figures
from asd_gen_gap.paper.export_tables import export_paper_tables


def test_paper_exports_run_without_evaluation_artifacts(tmp_path: Path) -> None:
    table = export_paper_tables(tmp_path)
    text = table.read_text(encoding="utf-8")
    assert "Internal bake-off results are not available." in text
    assert "No external validation performed" in text
    assert export_paper_figures(tmp_path) == []
