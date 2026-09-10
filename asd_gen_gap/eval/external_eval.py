"""Run one-shot external ABIDE II inference for fitted project models."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from asd_gen_gap.eval.one_shot import assert_one_shot_outputs
from asd_gen_gap.models.baseline import baseline_feature_columns


PREDICTION_COLUMNS = ("subject_id", "site", "dx_group", "y_prob", "y_pred", "model_name")
REQUIRED_COLUMNS = ("subject_id", "site", "dx_group")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"External data is missing required columns: {sorted(missing)}")
    labels = set(frame["dx_group"].dropna().unique())
    if not labels.issubset({0, 1}):
        raise ValueError("dx_group must be binary and encoded as 0 and 1")


def _positive_probabilities(model: object, values: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(values))  # type: ignore[attr-defined]
    classes = np.asarray(getattr(model, "classes_", []))
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes) or not np.any(classes == 1):
        raise ValueError("Model must provide predict_proba values for a class labelled 1")
    return probabilities[:, int(np.flatnonzero(classes == 1)[0])]


def _load_baseline(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"Missing fitted baseline model: {path}")
    try:
        import joblib

        model = joblib.load(path)
    except Exception:
        with path.open("rb") as handle:
            model = pickle.load(handle)
    if not hasattr(model, "predict_proba"):
        raise TypeError("Baseline model must be a fitted sklearn-style estimator with predict_proba")
    return model


def _hcan_checkpoint(checkpoint_dir: Path, fold: int) -> Path:
    matches = sorted(checkpoint_dir.glob(f"fold_{fold}_*.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one HCAN checkpoint for fold {fold} in {checkpoint_dir}; found {len(matches)}"
        )
    return matches[0]


def _predict_hcan(frame: pd.DataFrame, checkpoint_path: Path, *, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    import torch

    from asd_gen_gap.models.hcan import HCANModel, build_heterogeneous_graph, hcan_feature_columns

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or not {"state_dict", "config"}.issubset(checkpoint):
        raise ValueError(f"Invalid HCAN checkpoint: {checkpoint_path}")
    config = dict(checkpoint["config"])
    feature_cols = list(config.get("feature_cols", hcan_feature_columns(frame)))
    _require_columns(frame, feature_cols)
    model = HCANModel(
        input_dim=int(config["input_dim"]), hidden_size=int(config.get("hidden_size", 20)),
        num_layers=int(config.get("num_layers", 2)), dropout_rate=float(config.get("dropout_rate", 0.6)),
        edge_types=config.get("edge_types", ("sex", "handedness")), num_classes=int(config.get("num_classes", 2)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    graph = {name: edge.to(device) for name, edge in build_heterogeneous_graph(frame, model.config["edge_types"]).items()}
    features = torch.tensor(frame.loc[:, feature_cols].to_numpy(dtype=np.float32), device=device)
    with torch.no_grad():
        probabilities = torch.softmax(model(features, graph), dim=1).cpu().numpy()
    class_values = np.asarray(config.get("class_values", [0, 1]))
    if probabilities.shape[1] != len(class_values) or not np.any(class_values == 1):
        raise ValueError("HCAN checkpoint must define a class labelled 1")
    positive = probabilities[:, int(np.flatnonzero(class_values == 1)[0])]
    return positive, class_values[(positive >= 0.5).astype(int)]


def run_external_evaluation(model_name: Literal["baseline", "hcan"], fold: int, data_path: str | Path,
                            output_path: str | Path, *, override: bool = False,
                            baseline_model_path: str | Path | None = None,
                            checkpoint_dir: str | Path = "results/checkpoints/hcan",
                            device: str = "cpu") -> pd.DataFrame:
    """Generate standard-schema external predictions from a selected trained fold."""
    if fold < 0:
        raise ValueError("fold must be non-negative")
    output = Path(output_path)
    assert_one_shot_outputs([output], override=override)
    frame = pd.read_parquet(data_path)
    _require_columns(frame, REQUIRED_COLUMNS)
    if model_name == "baseline":
        path = Path(baseline_model_path) if baseline_model_path else Path("results/checkpoints/baseline") / f"fold_{fold}.joblib"
        model = _load_baseline(path)
        features = list(getattr(model, "feature_names_in_", baseline_feature_columns(frame)))
        _require_columns(frame, features)
        probabilities = _positive_probabilities(model, frame.loc[:, features])
        predicted = (probabilities >= 0.5).astype(int)
    elif model_name == "hcan":
        probabilities, predicted = _predict_hcan(frame, _hcan_checkpoint(Path(checkpoint_dir), fold), device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    predictions = pd.DataFrame({
        "subject_id": frame["subject_id"].to_numpy(), "site": frame["site"].to_numpy(),
        "dx_group": frame["dx_group"].to_numpy(), "y_prob": probabilities,
        "y_pred": predicted, "model_name": model_name,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output, index=False)
    return predictions


def evaluate_external_predictions(predictions_dir: str | Path = "results/predictions") -> pd.DataFrame:
    """Summarize already-written external predictions for both models."""
    rows: list[dict[str, object]] = []
    for model_name in ("baseline", "hcan"):
        path = Path(predictions_dir) / f"{model_name}_external.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing external predictions: {path}")
        predictions = pd.read_parquet(path)
        if missing := {"dx_group", "y_prob", "y_pred"}.difference(predictions.columns):
            raise ValueError(f"{path} is missing prediction columns: {sorted(missing)}")
        rows.append({"model_name": model_name, "n_subjects": len(predictions),
                     "accuracy": accuracy_score(predictions["dx_group"], predictions["y_pred"]),
                     "auc": roc_auc_score(predictions["dx_group"], predictions["y_prob"])})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("baseline", "hcan"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--data", required=True, help="ABIDE II parquet dataset")
    parser.add_argument("--out", required=True, help="Predictions parquet path")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--baseline-model", help="Optional fitted joblib/pickle baseline path")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints/hcan")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    run_external_evaluation(args.model, args.fold, args.data, args.out, override=args.override,
                            baseline_model_path=args.baseline_model, checkpoint_dir=args.checkpoint_dir, device=args.device)


if __name__ == "__main__":
    main()
