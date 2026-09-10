"""CPU smoke tests for the LOSO-safe HCAN adaptation."""

from pathlib import Path

import pandas as pd
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from asd_gen_gap.eval.loso import run_loso_hcan
from asd_gen_gap.models.hcan import build_heterogeneous_graph


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


def test_hcan_graph_and_loso_cpu_checkpoint_smoke(tmp_path: Path) -> None:
    frame = _fixture()
    graph = build_heterogeneous_graph(frame)
    assert set(graph) == {"sex", "handedness"}
    assert graph["sex"].shape[0] == graph["handedness"].shape[0] == 2
    assert graph["sex"].shape[1] > 0 and graph["handedness"].shape[1] > 0

    predictions = run_loso_hcan(frame, epochs=2, device="cpu", checkpoint_dir=tmp_path, hidden_size=4)
    assert len(predictions) == len(frame)
    assert predictions["subject_id"].is_unique
    assert set(predictions["model_name"]) == {"hcan"}
    checkpoints = list(tmp_path.glob("*.pt"))
    assert len(checkpoints) == frame["site"].nunique()
    checkpoint = torch.load(checkpoints[0], weights_only=True)
    assert {"state_dict", "config", "seed", "metrics"}.issubset(checkpoint)


def test_hcan_graph_supports_site_edges() -> None:
    graph = build_heterogeneous_graph(_fixture(), ("sex", "handedness", "site"))
    assert "site" in graph
    assert graph["site"].shape[1] > 0
