from pathlib import Path

import pandas as pd
import pytest

from asd_gen_gap.data.abide_loader import load_abide_i
from asd_gen_gap.data.phenotype import filter_by_motion, normalize_phenotypes


def _config(abide_i: Path) -> dict:
    return {"dataset_paths": {"abide_i": str(abide_i)}}


def _write_abide_fixture(root: Path, subject_ids: list[str]) -> None:
    pd.DataFrame(
        {
            "SUB_ID": subject_ids,
            "SITE_ID": ["SITE_A"] * len(subject_ids),
            "DX_GROUP": [1] * len(subject_ids),
            "AGE_AT_SCAN": [10.0 + index for index in range(len(subject_ids))],
            "SEX": [1] * len(subject_ids),
            "FIQ": [100.0 + index for index in range(len(subject_ids))],
            "func_mean_fd": [0.1, 0.3][: len(subject_ids)],
        }
    ).to_csv(root / "Phenotypic_V1_0b_preprocessed1.csv", index=False)


def test_load_abide_i_standardizes_columns_and_locates_timeseries(tmp_path: Path) -> None:
    _write_abide_fixture(tmp_path, ["1001", "1002"])
    for subject_id in ("1001", "1002"):
        (tmp_path / f"{subject_id}_rois_cc200.1D").write_text("1 2\n3 4\n")

    loaded = load_abide_i(_config(tmp_path))

    assert list(loaded.columns) == [
        "subject_id", "site", "dx_group", "age", "sex", "fiq", "func_mean_fd", "timeseries_path",
    ]
    assert loaded.loc[0, "subject_id"] == "1001"
    assert Path(loaded.loc[1, "timeseries_path"]).name == "1002_rois_cc200.1D"


def test_load_abide_i_reports_missing_timeseries_subjects(tmp_path: Path) -> None:
    _write_abide_fixture(tmp_path, ["1001", "1002"])
    (tmp_path / "1001_rois_cc200.1D").write_text("1 2\n")

    with pytest.raises(FileNotFoundError, match=r"1 subject\(s\): 1002"):
        load_abide_i(_config(tmp_path))


def test_normalization_and_subject_level_motion_filtering() -> None:
    source = pd.DataFrame(
        {
            "subject_id": ["a", "b", "c"],
            "site": ["one", "one", "two"],
            "dx_group": ["1", "2", "1"],
            "age": [10, 14, 12],
            "fiq": [90, 110, 100],
            "func_mean_fd": [0.2, 0.201, 0.1],
        }
    )

    normalized = normalize_phenotypes(source)
    filtered, report = filter_by_motion(normalized, threshold=0.2)

    assert normalized["dx_group"].dtype.kind in "iu"
    assert normalized.loc[0, "age_site_z"] == pytest.approx(-0.70710678)
    assert normalized.loc[1, "fiq_site_z"] == pytest.approx(0.70710678)
    assert filtered["subject_id"].tolist() == ["a", "c"]
    assert report["excluded_subject_ids"] == ["b"]
    assert report["method"] == "summary-level subject exclusion (not frame-level scrubbing)"
