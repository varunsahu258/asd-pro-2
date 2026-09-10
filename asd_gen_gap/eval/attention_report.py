"""Export HCAN semantic-attention and training-history figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from asd_gen_gap.models.hcan import _tensors, build_heterogeneous_graph, load_hcan_from_checkpoint

if TYPE_CHECKING:
    from asd_gen_gap.models.hcan import HCANModel


def extract_attention_weights(model: HCANModel, inference_df: pd.DataFrame, feature_cols: list[str],
                              edge_types: tuple[str, ...] | list[str], device: str = "cpu") -> dict[int, dict[str, float]]:
    """Run inference once and return each HCAN layer's semantic attention weights."""
    import torch

    active_device = next(model.parameters()).device
    inference = inference_df.copy()
    inference["_hcan_label"] = 0
    _, _, graph = _tensors(inference, feature_cols, edge_types, active_device)
    features = torch.tensor(inference.loc[:, feature_cols].to_numpy(dtype=np.float32), device=active_device)
    model.eval()
    with torch.no_grad():
        model(features, graph)
    return {
        index: {edge_type: float(weight) for edge_type, weight in zip(layer.edge_types, layer.attention.last_weights, strict=True)}
        for index, layer in enumerate(model.layers)
    }


def export_attention_figure(checkpoint_path: str | Path, inference_data_path: str | Path, *,
                            output_path: str | Path = "results/figure_hcan_attention_weights.png",
                            device: str = "cpu") -> Path:
    """Write a grouped bar chart of attention weights stored by one inference pass."""
    model, feature_cols = load_hcan_from_checkpoint(checkpoint_path, device=device)
    inference = pd.read_parquet(inference_data_path)
    weights = extract_attention_weights(model, inference, feature_cols, model.config["edge_types"], device=device)
    import matplotlib.pyplot as plt

    edge_types = list(model.config["edge_types"])
    width = 0.8 / len(weights)
    positions = np.arange(len(edge_types))
    figure, axis = plt.subplots(figsize=(max(5, len(edge_types) * 1.2), 4))
    for index, layer_weights in weights.items():
        offset = positions - 0.4 + width / 2 + index * width
        axis.bar(offset, [layer_weights[edge_type] for edge_type in edge_types], width=width, label=f"Layer {index + 1}")
    axis.set_xticks(positions, edge_types)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Attention weight")
    axis.set_title("HCAN meta-path attention weights")
    axis.legend()
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=300)
    plt.close(figure)
    return destination


def export_training_curve(checkpoint_path: str | Path,
                          output_path: str | Path = "results/figure_hcan_training_curves.png") -> Path:
    """Write a training-loss curve from the history retained in a checkpoint."""
    import torch
    import matplotlib.pyplot as plt

    load_hcan_from_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
    loss_history = metrics.get("loss_history")
    if not loss_history:
        raise ValueError(f"Checkpoint does not contain loss_history: {checkpoint_path}")
    figure, axis = plt.subplots(figsize=(5, 4))
    epochs = range(1, len(loss_history) + 1)
    axis.plot(epochs, loss_history, label="Training")
    if val_loss_history := metrics.get("val_loss_history"):
        axis.plot(range(1, len(val_loss_history) + 1), val_loss_history, label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title("HCAN training curve")
    axis.legend()
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=300)
    plt.close(figure)
    return destination


def main(argv: list[str] | None = None) -> None:
    """Export an HCAN attention figure or training curve from a checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="results/figure_hcan_attention_weights.png")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("attention", "training_curve"), default="attention")
    args = parser.parse_args(argv)
    if args.mode == "attention":
        export_attention_figure(args.checkpoint, args.data, output_path=args.out, device=args.device)
    else:
        export_training_curve(args.checkpoint, output_path=args.out)


if __name__ == "__main__":
    main()
