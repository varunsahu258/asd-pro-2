from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from asd_gen_gap.data.build_dataset import build_dataset, main


def _write_config(tmp_path: Path, abide_i: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"dataset_paths": {"abide_i": str(abide_i)}, "motion": {"max_mean_fd": 0.2}}))
    return path


def _write_cohort(root: Path) -> None:
    pd.DataFrame({
        "SUB_ID": ["one", "two", "three"], "SITE_ID": ["A", "B", "B"],
        "DX_GROUP": [1, 2, 1], "AGE_AT_SCAN": [10, 11, 12], "SEX": [1, 2, 1],
        "FIQ": [100, 101, 102], "func_mean_fd": [0.1, 0.1, 0.3],
    }).to_csv(root / "Phenotypic_V1_0b_preprocessed1.csv", index=False)
    for subject_id in ("one", "two", "three"):
        np.savetxt(root / f"{subject_id}_rois_cc200.1D", [[1, 2, 3], [2, 1, 2], [3, 3, 1]])


def test_build_dataset_end_to_end_and_site_filtering(tmp_path: Path) -> None:
    _write_cohort(tmp_path)
    config = _write_config(tmp_path, tmp_path)
    output = tmp_path / "result.parquet"

    result = build_dataset("I", output, config, sites=["B"])
    persisted = pd.read_parquet(output)

    expected_columns = {"subject_id", "dataset", "site", "dx_group", "age_site_z", "sex", "fiq_site_z", "conn_0_1", "conn_0_2", "conn_1_2"}
    assert len(result) == 1
    assert result["subject_id"].tolist() == ["two"]
    assert set(persisted.columns) == expected_columns


def test_build_dataset_rejects_requested_site_without_subjects(tmp_path: Path) -> None:
    _write_cohort(tmp_path)
    config = _write_config(tmp_path, tmp_path)

    with pytest.raises(ValueError, match="Requested site has zero matching subjects: absent"):
        main(["--dataset", "I", "--out", str(tmp_path / "result.parquet"), "--config", str(config), "--sites", "absent"])
