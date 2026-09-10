"""Tests for repeated-seed HCAN LOSO metric aggregation."""

from pathlib import Path

import pandas as pd
import pytest

from asd_gen_gap.eval.multiseed import aggregate_seed_runs


def _predictions(seed: int) -> pd.DataFrame:
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    probabilities = [0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6]
    if seed == 1:
        probabilities[2] = 0.6
    elif seed == 2:
        probabilities[5] = 0.4
    return pd.DataFrame({
        "subject_id": [f"subject-{index}" for index in range(8)],
        "site": ["one"] * 4 + ["two"] * 4,
        "dx_group": labels,
        "y_prob": probabilities,
        "y_pred": [int(value >= 0.5) for value in probabilities],
        "model_name": "hcan",
    })


def test_aggregate_seed_runs_reports_population_variance(tmp_path: Path) -> None:
    paths = []
    for seed in range(3):
        path = tmp_path / f"hcan_internal_seed{seed}.parquet"
        _predictions(seed).to_parquet(path, index=False)
        paths.append(path)

    summary = aggregate_seed_runs(paths, model_name="hcan")

    assert (summary["n_seeds"] == 3).all()
    assert (summary["accuracy_std"] >= 0).all()
    assert (summary["auc_std"] >= 0).all()
    assert (summary["scope"] == "pooled").sum() == 1


def test_aggregate_seed_runs_rejects_mismatched_subjects(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _predictions(0).to_parquet(first, index=False)
    changed = _predictions(1)
    changed.loc[0, "subject_id"] = "different-subject"
    changed.to_parquet(second, index=False)

    with pytest.raises(ValueError, match="identical subject_id"):
        aggregate_seed_runs([first, second], model_name="hcan")
