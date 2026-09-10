"""Aggregate variance metrics across independently repeated HCAN LOSO runs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from asd_gen_gap.eval.bakeoff_table import _markdown


REQUIRED_COLUMNS = ("subject_id", "site", "dx_group", "y_prob", "y_pred", "model_name")


def _validate_predictions(path: str | Path, model_name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if missing := set(REQUIRED_COLUMNS).difference(frame.columns):
        raise ValueError(f"{path} is missing prediction columns: {sorted(missing)}")
    if not frame["subject_id"].is_unique:
        raise ValueError(f"{path} must have one prediction per subject")
    if set(frame["model_name"].dropna().unique()) != {model_name}:
        raise ValueError(f"{path} must have model_name={model_name!r}")
    return frame


def _run_metrics(frame: pd.DataFrame, model_name: str) -> list[dict[str, object]]:
    groups = [("pooled", "pooled", frame)]
    groups.extend(("site", site, subset) for site, subset in frame.groupby("site", sort=True))
    rows: list[dict[str, object]] = []
    for scope, site, subset in groups:
        auc = float("nan")
        if subset["dx_group"].nunique() == 2:
            auc = float(roc_auc_score(subset["dx_group"], subset["y_prob"]))
        rows.append({
            "scope": scope, "site": site, "model_name": model_name,
            "accuracy": float(accuracy_score(subset["dx_group"], subset["y_pred"])), "auc": auc,
        })
    return rows


def aggregate_seed_runs(paths: Sequence[str | Path], *, model_name: str) -> pd.DataFrame:
    """Summarize pooled and per-site metrics across repeated seed runs."""
    if not paths:
        raise ValueError("paths must contain at least one prediction parquet")
    metric_rows: list[dict[str, object]] = []
    reference_labels: pd.Series | None = None
    for path in paths:
        frame = _validate_predictions(path, model_name)
        labels = frame.set_index("subject_id")["dx_group"].sort_index()
        if reference_labels is None:
            reference_labels = labels
        elif not labels.equals(reference_labels):
            raise ValueError("Seed prediction runs must have identical subject_id and dx_group mappings")
        metric_rows.extend(_run_metrics(frame, model_name))

    metrics = pd.DataFrame(metric_rows)
    rows: list[dict[str, object]] = []
    for (scope, site, name), group in metrics.groupby(["scope", "site", "model_name"], sort=True):
        rows.append({
            "scope": scope, "site": site, "model_name": name, "n_seeds": len(paths),
            "accuracy_mean": group["accuracy"].mean(), "accuracy_std": group["accuracy"].std(ddof=0),
            "auc_mean": group["auc"].mean(), "auc_std": group["auc"].std(ddof=0),
        })
    return pd.DataFrame(rows, columns=[
        "scope", "site", "model_name", "n_seeds", "accuracy_mean", "accuracy_std", "auc_mean", "auc_std",
    ])


def main(argv: list[str] | None = None) -> None:
    """Write a Markdown seed-stability summary from repeated LOSO predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", action="append", required=True, help="Seed prediction parquet (repeatable)")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out", required=True, help="Output Markdown path")
    args = parser.parse_args(argv)
    if len(args.predictions) < 2:
        parser.error("--predictions must be provided at least twice")
    table = aggregate_seed_runs(args.predictions, model_name=args.model_name)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("# Seed stability\n\n" + _markdown(table) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
