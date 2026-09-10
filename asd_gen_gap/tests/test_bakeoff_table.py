"""Tests for model-discovering internal bake-off tables."""

from pathlib import Path

import pandas as pd

from asd_gen_gap.eval.bakeoff_table import build_bakeoff_table


def _predictions(model_name: str, offset: float) -> pd.DataFrame:
    rows = []
    for site_index, site in enumerate(("one", "two")):
        for subject_index, label in enumerate((0, 1, 0, 1)):
            probability = 0.2 + 0.5 * label + 0.03 * subject_index + offset + 0.01 * site_index
            rows.append({
                "subject_id": f"{site}-{subject_index}", "site": site, "dx_group": label,
                "y_prob": min(probability, 0.99), "y_pred": int(probability >= 0.5),
                "model_name": model_name,
            })
    return pd.DataFrame(rows)


def test_build_bakeoff_table_discovers_all_models_and_corrects_jointly(tmp_path: Path) -> None:
    for model_name, offset in (("baseline", 0.0), ("hcan", 0.02), ("hcan_site", -0.02)):
        _predictions(model_name, offset).to_parquet(tmp_path / f"{model_name}_internal.parquet", index=False)

    table = build_bakeoff_table(tmp_path, n_resamples=20)

    assert len(table.loc[table["scope"] == "pooled"]) == 3
    assert len(table.loc[table["scope"] == "site"]) == 3 * 2
    pairwise = table.attrs["pairwise_tests"]
    assert len(pairwise) == 6
    assert (pairwise.loc[pairwise["test"] == "AUC delta (paired bootstrap)", "auc_delta_ci_low"].notna()).all()
    ordered = pairwise.dropna(subset=["p_value"]).sort_values("p_value")
    assert (ordered["p_value_holm"].to_numpy() >= ordered["p_value"].to_numpy()).all()
