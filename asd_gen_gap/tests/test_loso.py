"""Tests for leave-one-site-out prediction generation."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from asd_gen_gap.eval.loso import run_loso
from asd_gen_gap.models.baseline import baseline_feature_columns, make_baseline_model


def synthetic_dataset() -> pd.DataFrame:
    """Small in-memory stand-in for the expected parquet dataset."""
    rows = []
    for site_index, site in enumerate(("A", "B", "C")):
        for subject_offset, diagnosis in enumerate((0, 1, 0, 1)):
            rows.append(
                {
                    "subject_id": f"{site}-{subject_offset}",
                    "site": site,
                    "dx_group": diagnosis,
                    "conn_0_1": diagnosis * 2 + site_index * 0.1,
                    "conn_0_2": diagnosis - site_index * 0.1,
                    "age_site_z": subject_offset - 1.5,
                    "sex": subject_offset % 2,
                    "fiq_site_z": 1.5 - subject_offset,
                }
            )
    return pd.DataFrame(rows)


def test_run_loso_holds_out_every_site_exactly_once():
    dataset = synthetic_dataset()
    features = baseline_feature_columns(dataset)

    predictions = run_loso(dataset, make_baseline_model, features)

    assert len(predictions) == len(dataset)
    assert predictions["subject_id"].is_unique
    assert set(predictions["site"]) == set(dataset["site"])
    assert predictions.groupby("site").size().to_dict() == dataset.groupby("site").size().to_dict()
    assert predictions["y_prob"].between(0, 1).all()
    assert set(predictions["y_pred"]).issubset({0, 1})


def test_run_loso_accepts_any_sklearn_probability_estimator():
    dataset = synthetic_dataset()
    predictions = run_loso(
        dataset,
        lambda: LogisticRegression(max_iter=1_000),
        ["conn_0_1", "conn_0_2"],
    )
    assert set(predictions["model_name"]) == {"LogisticRegression"}


def test_run_loso_hcan_uses_custom_model_name_with_site_edges(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from asd_gen_gap.eval.loso import run_loso_hcan

    frame = synthetic_dataset()
    frame["handedness"] = ["R", "R", "L", "L"] * 3
    predictions = run_loso_hcan(
        frame, epochs=2, device="cpu", checkpoint_dir=tmp_path, hidden_size=4,
        edge_types=("sex", "handedness", "site"), model_name="hcan_site",
    )
    assert set(predictions["model_name"]) == {"hcan_site"}
