"""Tests for HCAN semantic-attention reporting."""

import pandas as pd
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from asd_gen_gap.eval.attention_report import extract_attention_weights
from asd_gen_gap.models.hcan import hcan_feature_columns, train_hcan


def _fixture() -> pd.DataFrame:
    rows = []
    for site, offset in (("one", 0), ("two", 1), ("three", 2)):
        for index, label in enumerate((0, 1, 0, 1)):
            rows.append({
                "subject_id": f"{site}-{index}", "site": site, "dx_group": label,
                "conn_0_1": label + offset * 0.01, "conn_0_2": (1 - label) + offset * 0.02,
                "age_site_z": index - 1.5, "sex": index % 2,
                "fiq_site_z": 1.5 - index, "handedness": "R" if index < 2 else "L",
            })
    return pd.DataFrame(rows)


def test_extract_attention_weights_sum_to_one_per_layer() -> None:
    frame = _fixture()
    model = train_hcan(frame, epochs=2, device="cpu", hidden_size=4)

    weights = extract_attention_weights(model, frame, hcan_feature_columns(frame), model.config["edge_types"])

    assert len(weights) == len(model.layers)
    for layer_weights in weights.values():
        assert set(layer_weights) == set(model.config["edge_types"])
        assert sum(layer_weights.values()) == pytest.approx(1.0, abs=1e-5)
