"""Sex-stratified performance summaries from existing prediction artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from asd_gen_gap.eval.bakeoff_table import _auc, _markdown, _sensitivity, _specificity


DEFAULT_DATASETS = {"internal": "results/abide_i_cc200.parquet", "external": "results/abide_ii_external.parquet"}


def build_subgroup_report(predictions_dir: str | Path = "results/predictions",
                          dataset_paths: dict[str, str | Path] | None = None, *, min_group_n: int = 10) -> pd.DataFrame:
    """Return sex-stratified metrics for all discoverable internal/external predictions."""
    paths = {**DEFAULT_DATASETS, **(dataset_paths or {})}
    rows: list[dict[str, object]] = []
    directory = Path(predictions_dir)
    for scope in ("internal", "external"):
        dataset_path = Path(paths[scope])
        if not dataset_path.is_file():
            continue
        source = pd.read_parquet(dataset_path)
        if {"subject_id", "sex"}.difference(source.columns):
            raise ValueError(f"{dataset_path} must contain subject_id and sex")
        for path in sorted(directory.glob(f"*_{scope}.parquet")):
            model_name = path.name.removesuffix(f"_{scope}.parquet")
            frame = pd.read_parquet(path).merge(source.loc[:, ["subject_id", "sex"]], on="subject_id", how="left", validate="one_to_one")
            for sex, group in frame.groupby("sex", dropna=False):
                n_subjects = len(group)
                rows.append({"model_name": model_name, "scope": scope, "sex": sex, "n_subjects": n_subjects,
                             "accuracy": float(accuracy_score(group["dx_group"], group["y_pred"])), "auc": _auc(group),
                             "sensitivity": _sensitivity(group), "specificity": _specificity(group),
                             "coverage": "adequate" if n_subjects >= min_group_n else "indicative"})
    return pd.DataFrame(rows, columns=["model_name", "scope", "sex", "n_subjects", "accuracy", "auc", "sensitivity", "specificity", "coverage"])


def write_subgroup_report(predictions_dir: str | Path = "results/predictions", dataset_paths: dict[str, str | Path] | None = None, *,
                          output_path: str | Path = "results/table_subgroup_by_sex.md", min_group_n: int = 10) -> pd.DataFrame:
    """Write and return the sex-stratified performance table."""
    report = build_subgroup_report(predictions_dir, dataset_paths, min_group_n=min_group_n)
    destination = Path(output_path); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("# Sex-stratified performance\n\n" + _markdown(report) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", default="results/predictions"); parser.add_argument("--out", default="results/table_subgroup_by_sex.md")
    parser.add_argument("--internal-data"); parser.add_argument("--external-data")
    args = parser.parse_args(argv)
    paths = {key: value for key, value in {"internal": args.internal_data, "external": args.external_data}.items() if value}
    write_subgroup_report(args.predictions_dir, paths or None, output_path=args.out)


if __name__ == "__main__":
    main()
