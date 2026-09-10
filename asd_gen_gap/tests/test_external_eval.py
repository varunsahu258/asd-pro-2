"""Tests for external prediction production and one-shot output protection."""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from asd_gen_gap.eval.external_eval import run_external_evaluation
from asd_gen_gap.eval.one_shot import assert_one_shot_outputs
from asd_gen_gap.models.baseline import baseline_feature_columns, make_baseline_model


def _fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": ["s1", "s2", "s3", "s4"], "site": ["external"] * 4,
        "dx_group": [0, 1, 0, 1], "conn_0_1": [-2.0, 2.0, -1.0, 1.0],
        "age_site_z": [-1.0, 1.0, -0.5, 0.5], "sex": [0, 1, 0, 1],
        "fiq_site_z": [1.0, -1.0, 0.5, -0.5],
    })


def test_one_shot_guard_rejects_existing_outputs(tmp_path: Path) -> None:
    output = tmp_path / "existing.parquet"
    output.touch()
    with pytest.raises(FileExistsError, match="override=True"):
        assert_one_shot_outputs([output])
    assert_one_shot_outputs([output], override=True)


def test_external_baseline_evaluation_writes_standard_predictions(tmp_path: Path) -> None:
    frame = _fixture()
    data = tmp_path / "abide_ii.parquet"
    frame.to_parquet(data, index=False)
    model = make_baseline_model().fit(frame.loc[:, baseline_feature_columns(frame)], frame["dx_group"])
    trained = tmp_path / "fold_3.joblib"
    joblib.dump(model, trained)
    output = tmp_path / "baseline_external.parquet"

    predictions = run_external_evaluation("baseline", 3, data, output, baseline_model_path=trained)

    assert output.is_file()
    assert list(predictions.columns) == ["subject_id", "site", "dx_group", "y_prob", "y_pred", "model_name"]
    assert predictions["y_prob"].between(0, 1).all()
    assert set(predictions["model_name"]) == {"baseline"}
    with pytest.raises(FileExistsError):
        run_external_evaluation("baseline", 3, data, output, baseline_model_path=trained)


def test_external_hcan_evaluation_loads_loso_format_checkpoint_on_cpu(tmp_path: Path) -> None:
    """External HCAN inference accepts the checkpoint dictionary written by LOSO."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from asd_gen_gap.models.hcan import train_hcan

    frame = _fixture().assign(handedness=["R", "L", "R", "L"])
    data = tmp_path / "abide_ii.parquet"
    frame.to_parquet(data, index=False)
    model = train_hcan(frame, epochs=2, device="cpu", hidden_size=4, seed=11)
    checkpoint_dir = tmp_path / "results" / "checkpoints" / "hcan"
    checkpoint_dir.mkdir(parents=True)
    # This is intentionally the same dictionary schema saved in run_loso_hcan.
    checkpoint = checkpoint_dir / "fold_5_external.pt"
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config, "seed": 11,
         "metrics": model.training_metrics},
        checkpoint,
    )
    output = tmp_path / "hcan_external.parquet"

    predictions = run_external_evaluation(
        "hcan", 5, data, output, checkpoint_dir=checkpoint_dir, device="cpu"
    )

    assert output.is_file()
    assert list(predictions.columns) == ["subject_id", "site", "dx_group", "y_prob", "y_pred", "model_name"]
    assert predictions["y_prob"].between(0, 1).all()
    assert set(predictions["y_pred"]).issubset({0, 1})
    assert set(predictions["model_name"]) == {"hcan"}
