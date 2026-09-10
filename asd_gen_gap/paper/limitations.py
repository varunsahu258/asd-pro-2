"""Compose data-driven manuscript limitations text."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


MIN_SITE_N = 30


def _read_gap_report(results_dir: Path) -> pd.DataFrame | None:
    path = results_dir / "gap_report.csv"
    return pd.read_csv(path) if path.is_file() else None


def generate_limitations(results_dir: str | Path = "results", *, min_site_n: int = MIN_SITE_N) -> str:
    """Generate a coherent limitations section from available report artifacts."""
    results = Path(results_dir)
    paragraphs = [
        "ABIDE I used summary-level mean framewise-displacement exclusion rather than frame-level scrubbing; this simplification should be kept in mind when interpreting motion-related comparability.",
    ]
    gap = _read_gap_report(results)
    if gap is not None and "external_validation_status" in gap and (gap["external_validation_status"] == "no external validation performed").any():
        paragraphs.append("No external validation was performed, so no internal/external comparison should be presented as conclusive.")
    elif gap is not None:
        wide_models = gap.loc[gap.get("ci_width_flag", pd.Series(index=gap.index, dtype=str)) == "wide", "model_name"].dropna().tolist()
        if wide_models:
            paragraphs.append(f"The external comparison for {', '.join(wide_models)} is imprecise because its external confidence interval is wide and should not be treated as conclusive.")
        counts: dict[str, int] = {}
        if "external_site_counts" in gap:
            for encoded in gap["external_site_counts"].dropna():
                counts.update({site: int(number) for site, number in json.loads(encoded).items()})
        small_sites = [f"{site} (n={count})" for site, count in sorted(counts.items()) if count < min_site_n]
        if small_sites:
            paragraphs.append("Per-site external estimates for " + ", ".join(small_sites) + f" are indicative rather than conclusive because these sites fall below the prespecified minimum of n={min_site_n}.")
    manifest = results / "abide_ii_preproc_manifest.csv"
    if manifest.is_file():
        paragraphs.append("ABIDE I used C-PAC, whereas ABIDE II used a custom minimal pipeline with affine-only normalization, motion plus tissue-mean regression, and no FreeSurfer segmentation. This is a genuine confound in a site-generalization study: some observed gap may reflect pipeline differences rather than pure site effects, although it is a caveat rather than a disqualifying flaw.")
    note_path = results / "hcan_adaptation_notes.md"
    if note_path.is_file():
        paragraphs.append("The HCAN component is adapted from Shao, Fu & Chen (2023), with its site meta-path removed; its 10-fold mixed-site cross-validation accuracy is not directly comparable with this project's LOSO plus external-validation protocol.")
    return "\n\n".join(paragraphs)


def write_limitations(output_path: str | Path, results_dir: str | Path = "results") -> str:
    """Write and return generated limitations text."""
    text = generate_limitations(results_dir)
    Path(output_path).write_text(text + "\n", encoding="utf-8")
    return text
