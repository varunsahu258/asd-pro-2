"""Dataset-agnostic leave-one-site-out evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("subject_id", "site", "dx_group")


def _positive_probabilities(model: object, values: pd.DataFrame) -> np.ndarray:
    """Extract positive-class probabilities from a fitted sklearn-like model."""
    probabilities = np.asarray(model.predict_proba(values))  # type: ignore[attr-defined]
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("model.predict_proba must return probabilities for two classes")
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    positive_index = int(np.where(classes == 1)[0][0]) if np.any(classes == 1) else 1
    return probabilities[:, positive_index]


def run_loso(
    df: pd.DataFrame,
    model_fn: Callable[[], object],
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Fit one model per held-out site and return subject-level predictions.

    ``model_fn`` must construct a fresh estimator implementing ``fit`` and
    ``predict_proba``. Binary labels must be encoded as 0/1, with 1 denoting
    the class whose probability is reported in ``y_prob``.
    """
    missing = set(REQUIRED_COLUMNS).union(feature_cols).difference(df.columns)
    if missing:
        raise ValueError(f"Dataframe is missing required columns: {sorted(missing)}")
    if not feature_cols:
        raise ValueError("feature_cols must not be empty")
    if df["site"].isna().any() or df["subject_id"].isna().any():
        raise ValueError("site and subject_id cannot contain missing values")
    labels = set(pd.unique(df["dx_group"]))
    if not labels.issubset({0, 1}):
        raise ValueError("dx_group must be binary and encoded as 0 and 1")

    frames: list[pd.DataFrame] = []
    for site in pd.unique(df["site"]):
        test_mask = df["site"] == site
        train, test = df.loc[~test_mask], df.loc[test_mask]
        if train.empty or test.empty:
            raise ValueError("LOSO requires observations both inside and outside every site")
        if train["dx_group"].nunique() < 2:
            raise ValueError(f"Training data for held-out site {site!r} has only one class")

        model = model_fn()
        model.fit(train.loc[:, feature_cols], train["dx_group"])
        y_prob = _positive_probabilities(model, test.loc[:, feature_cols])
        frames.append(
            pd.DataFrame(
                {
                    "subject_id": test["subject_id"].to_numpy(),
                    "site": test["site"].to_numpy(),
                    "dx_group": test["dx_group"].to_numpy(),
                    "y_prob": y_prob,
                    "y_pred": (y_prob >= 0.5).astype(int),
                    "model_name": getattr(model, "model_name", model.__class__.__name__),
                },
                index=test.index,
            )
        )
    return pd.concat(frames).sort_index().reset_index(drop=True)


def run_loso_hcan(
    df: pd.DataFrame,
    *,
    epochs: int = 200,
    device: str = "cuda",
    checkpoint_dir: str | Path = "results/checkpoints/hcan",
    seed: int = 0,
    **train_kwargs: object,
) -> pd.DataFrame:
    """Run LOSO using the sex/handedness-only HCAN adaptation.

    HCAN is adapted from Shao, Fu & Chen (2023). Its original site meta-path is
    deliberately absent: at inference each held-out subject is appended to the
    training graph solely through sex and handedness, never site identity.
    """
    # Keep torch-geometric optional for users of the non-neural baselines.
    import torch

    from asd_gen_gap.models.hcan import build_heterogeneous_graph, hcan_feature_columns, train_hcan

    required = {"subject_id", "site", "dx_group", "sex", "handedness"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataframe is missing required HCAN columns: {sorted(missing)}")
    feature_cols = hcan_feature_columns(df)
    destination = Path(checkpoint_dir)
    destination.mkdir(parents=True, exist_ok=True)
    prediction_frames: list[pd.DataFrame] = []
    for fold_number, site in enumerate(pd.unique(df["site"])):
        test_mask = df["site"] == site
        train, test = df.loc[~test_mask].copy(), df.loc[test_mask].copy()
        if train["dx_group"].nunique() != 2:
            raise ValueError(f"Training data for held-out site {site!r} has fewer than two classes")
        model = train_hcan(train, epochs=epochs, device=device, seed=seed + fold_number, **train_kwargs)
        active_device = next(model.parameters()).device
        # This graph extends, rather than replaces, the fitted training graph.
        # Its construction is restricted by hcan.py to sex and handedness.
        inference = pd.concat([train, test], ignore_index=True)
        graph = {name: edge.to(active_device) for name, edge in build_heterogeneous_graph(inference, model.config["edge_types"]).items()}
        features = torch.tensor(inference.loc[:, feature_cols].to_numpy(dtype=np.float32), device=active_device)
        model.eval()
        with torch.no_grad():
            probabilities = torch.softmax(model(features, graph), dim=1).cpu().numpy()
        test_probabilities = probabilities[len(train):, 1]
        class_values = np.asarray(model.class_values)
        predicted = class_values[(test_probabilities >= 0.5).astype(int)]
        metrics = {**model.training_metrics, "held_out_site": str(site), "n_train": len(train), "n_test": len(test)}
        torch.save(
            {"state_dict": model.state_dict(), "config": model.config, "seed": seed + fold_number, "metrics": metrics},
            destination / f"fold_{fold_number}_{str(site)}.pt",
        )
        prediction_frames.append(pd.DataFrame({
            "subject_id": test["subject_id"].to_numpy(), "site": test["site"].to_numpy(),
            "dx_group": test["dx_group"].to_numpy(), "y_prob": test_probabilities,
            "y_pred": predicted, "model_name": "hcan",
        }, index=test.index))
    return pd.concat(prediction_frames).sort_index().reset_index(drop=True)
