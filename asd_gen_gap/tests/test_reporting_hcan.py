"""Regression tests for HCAN-named reporting artifacts and limitations text."""

from pathlib import Path

import pandas as pd

from asd_gen_gap.eval.bakeoff_table import build_bakeoff_table, write_bakeoff_table
from asd_gen_gap.eval.external_eval import evaluate_external_predictions
from asd_gen_gap.paper.limitations import generate_limitations


def _predictions(model_name: str) -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": list("abcdefgh"), "site": ["one", "one", "two", "two"] * 2,
        "dx_group": [0, 1, 0, 1, 0, 1, 0, 1],
        "y_prob": [0.1, 0.9, 0.2, 0.8, 0.15, 0.7, 0.45, 0.55] if model_name == "baseline" else [0.2, 0.8, 0.4, 0.6, 0.35, 0.65, 0.3, 0.7],
        "y_pred": [0, 1, 0, 1, 0, 1, 0, 1], "model_name": [model_name] * 8,
    })


def test_bakeoff_and_external_evaluation_read_hcan_named_fixtures(tmp_path: Path) -> None:
    for split in ("internal", "external"):
        _predictions("baseline").to_parquet(tmp_path / f"baseline_{split}.parquet", index=False)
        _predictions("hcan").to_parquet(tmp_path / f"hcan_{split}.parquet", index=False)

    internal = write_bakeoff_table(tmp_path, output_path=tmp_path / "table_internal_bakeoff.md", n_resamples=100)
    external = evaluate_external_predictions(tmp_path)

    assert set(internal["model_name"]) == {"baseline", "hcan"}
    assert set(internal["scope"]) == {"pooled", "site"}
    assert len(internal) == 6  # two pooled rows and two models for each of two sites
    assert {"accuracy_ci_low", "accuracy_ci_high", "auc_ci_low", "auc_ci_high"}.issubset(internal.columns)
    assert set(internal.attrs["pairwise_tests"]["test"]) == {"DeLong AUC", "McNemar accuracy"}
    markdown = (tmp_path / "table_internal_bakeoff.md").read_text(encoding="utf-8")
    assert "# Internal model bake-off" in markdown
    assert "## Paired comparisons" in markdown
    assert external["model_name"].tolist() == ["baseline", "hcan"]
    assert internal.loc[internal["scope"] == "pooled", "accuracy"].tolist() == [1.0, 1.0]
    assert external["auc"].tolist() == [1.0, 1.0]


def test_limitations_includes_hcan_citation_only_when_note_exists(tmp_path: Path) -> None:
    without_note = generate_limitations(tmp_path)
    assert "Shao, Fu & Chen" not in without_note
    (tmp_path / "hcan_adaptation_notes.md").write_text("Local HCAN adaptation note.", encoding="utf-8")

    with_note = generate_limitations(tmp_path)
    assert "Shao, Fu & Chen (2023)" in with_note
    assert "site meta-path removed" in with_note
    assert "10-fold mixed-site cross-validation" in with_note
