from pathlib import Path

import pandas as pd

from asd_gen_gap.eval.subgroup_report import build_subgroup_report


def test_subgroup_report_flags_small_sex_groups(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"; predictions.mkdir()
    frame = pd.DataFrame({"subject_id": list("abcd"), "site": ["one"] * 4, "dx_group": [0, 1, 0, 1], "y_prob": [.1, .9, .2, .8], "y_pred": [0, 1, 0, 1], "model_name": ["hcan"] * 4})
    frame.to_parquet(predictions / "hcan_internal.parquet", index=False)
    data = tmp_path / "internal.parquet"; pd.DataFrame({"subject_id": list("abcd"), "sex": [0, 0, 1, 1]}).to_parquet(data, index=False)
    report = build_subgroup_report(predictions, {"internal": data}, min_group_n=3)
    assert len(report) == 2
    assert set(report["coverage"]) == {"indicative"}
