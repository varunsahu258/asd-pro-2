"""LOSO-safe adaptation of HCAN from Shao, Fu, and Chen (2023).

This adapts the heterogeneous graph convolutional attention network (HCAN) in
Shao, Fu & Chen, *BMC Bioinformatics* 24, 98 (2023),
doi:10.1186/s12859-023-05241-2.  The published model uses sex, handedness, and
site meta-paths.  This implementation deliberately removes the site meta-path:
site links would leak held-out-site identity in LOSO and cannot be constructed
for a genuinely new external subject.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


DEFAULT_EDGE_TYPES = ("sex", "handedness")
CLINICAL_FEATURES = ("age_site_z", "sex", "fiq_site_z")


def build_heterogeneous_graph(
    df: pd.DataFrame, edge_types: Sequence[str] = DEFAULT_EDGE_TYPES
) -> dict[str, Tensor]:
    """Build undirected phenotype adjacencies without ever using ``site``.

    Each returned ``edge_index`` has shape ``[2, E]`` and connects every pair
    of different rows sharing a categorical phenotype value. This is the
    sex/handedness-only HCAN adaptation described by Shao et al. (2023), so it
    works unchanged for training folds, train-plus-held-out inference graphs,
    and external cohorts.
    """
    if not edge_types:
        raise ValueError("edge_types must not be empty")
    missing = set(edge_types).difference(df.columns)
    if missing:
        raise ValueError(f"Missing graph phenotype columns: {sorted(missing)}")
    graph: dict[str, Tensor] = {}
    for edge_type in edge_types:
        # Treat absent categories explicitly rather than making every missing
        # value part of one accidental shared group.
        values = df[edge_type].astype("string").fillna("Unknown").to_numpy()
        sources: list[int] = []
        targets: list[int] = []
        for value in pd.unique(values):
            indices = np.flatnonzero(values == value)
            for source in indices:
                for target in indices:
                    if source != target:
                        sources.append(int(source))
                        targets.append(int(target))
        graph[edge_type] = torch.tensor([sources, targets], dtype=torch.long) if sources else torch.empty((2, 0), dtype=torch.long)
    return graph


class ResidualGCNLayer(nn.Module):
    """GCNConv propagation plus the residual projection M in Shao et al. (2023)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return F.relu(self.gcn(x, edge_index) + self.residual(x))


class SemanticAttention(nn.Module):
    """Shared transform and semantic attention fusion from HCAN (Shao et al., 2023)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.q = nn.Parameter(torch.empty(hidden_size))
        nn.init.xavier_uniform_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)
        nn.init.normal_(self.q)

    def forward(self, embeddings: Sequence[Tensor]) -> Tensor:
        if not embeddings:
            raise ValueError("SemanticAttention requires at least one meta-path embedding")
        stacked = torch.stack(list(embeddings), dim=1)  # nodes, meta-paths, hidden
        transformed = torch.tanh(self.transform(stacked))
        scores = torch.matmul(transformed.mean(dim=0), self.q)  # one shared score per meta-path
        weights = torch.softmax(scores, dim=0)
        self.last_weights = weights.detach().cpu()
        return (stacked * weights.view(1, -1, 1)).sum(dim=1)


class HCANLayer(nn.Module):
    """Per-meta-path residual GCNs followed by HCAN semantic fusion."""

    def __init__(self, in_channels: int, out_channels: int, edge_types: Sequence[str]) -> None:
        super().__init__()
        self.edge_types = tuple(edge_types)
        self.convolutions = nn.ModuleDict({edge_type: ResidualGCNLayer(in_channels, out_channels) for edge_type in self.edge_types})
        self.attention = SemanticAttention(out_channels)

    def forward(self, x: Tensor, edge_indices: Mapping[str, Tensor]) -> Tensor:
        missing = set(self.edge_types).difference(edge_indices)
        if missing:
            raise ValueError(f"Graph lacks edge types: {sorted(missing)}")
        return self.attention([self.convolutions[name](x, edge_indices[name]) for name in self.edge_types])


class HCANModel(nn.Module):
    """Two-layer, dropout-0.6 HCAN adaptation of Shao et al. (2023)."""

    def __init__(self, input_dim: int, hidden_size: int = 20, num_layers: int = 2,
                 dropout_rate: float = 0.6, edge_types: Sequence[str] = DEFAULT_EDGE_TYPES,
                 num_classes: int = 2) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        self.config = {"input_dim": input_dim, "hidden_size": hidden_size, "num_layers": num_layers,
                       "dropout_rate": dropout_rate, "edge_types": list(edge_types), "num_classes": num_classes}
        dimensions = [input_dim, *([hidden_size] * num_layers)]
        self.layers = nn.ModuleList(HCANLayer(dimensions[index], dimensions[index + 1], edge_types) for index in range(num_layers))
        self.dropout_rate = dropout_rate
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor, edge_indices: Mapping[str, Tensor]) -> Tensor:
        for layer in self.layers:
            x = F.dropout(layer(x, edge_indices), p=self.dropout_rate, training=self.training)
        return self.classifier(x)


def hcan_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select connectivity plus clinical features used as HCAN node attributes."""
    columns = sorted(column for column in df if column.startswith("conn_")) + list(CLINICAL_FEATURES)
    missing = set(columns).difference(df.columns)
    if missing or not any(column.startswith("conn_") for column in columns):
        raise ValueError(f"Missing HCAN feature columns: {sorted(missing) or ['conn_<i>_<j>']}")
    return columns


def load_hcan_from_checkpoint(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> tuple[HCANModel, list[str]]:
    """Reconstruct an evaluation-ready HCAN model from a saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or not {"state_dict", "config"}.issubset(checkpoint):
        raise ValueError(f"Invalid HCAN checkpoint: {checkpoint_path}")
    config = dict(checkpoint["config"])
    model = HCANModel(
        input_dim=int(config["input_dim"]), hidden_size=int(config.get("hidden_size", 20)),
        num_layers=int(config.get("num_layers", 2)), dropout_rate=float(config.get("dropout_rate", 0.6)),
        edge_types=config.get("edge_types", DEFAULT_EDGE_TYPES), num_classes=int(config.get("num_classes", 2)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.config = config
    feature_cols = list(config.get("feature_cols", []))
    model.feature_cols = feature_cols
    model.class_values = list(config.get("class_values", [0, 1]))
    model.eval()
    return model, feature_cols


def model_summary(model: HCANModel) -> dict[str, object]:
    """Return lightweight architecture metadata for evaluation checkpoints."""
    return {
        "trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "hidden_size": model.config["hidden_size"], "num_layers": model.config["num_layers"],
        "edge_types": model.config["edge_types"],
    }


def _device(device: str | torch.device) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA was requested for HCAN but is unavailable; falling back to CPU.", RuntimeWarning, stacklevel=2)
        return torch.device("cpu")
    return requested


def _tensors(df: pd.DataFrame, feature_cols: Sequence[str], edge_types: Sequence[str], device: torch.device) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    x = torch.tensor(df.loc[:, feature_cols].to_numpy(dtype=np.float32), device=device)
    labels = torch.tensor(df["_hcan_label"].to_numpy(dtype=np.int64), device=device)
    graph = {name: edge.to(device) for name, edge in build_heterogeneous_graph(df, edge_types).items()}
    return x, labels, graph


def train_hcan(train_df: pd.DataFrame, val_df: pd.DataFrame | None = None, *, epochs: int = 200,
               device: str | torch.device = "cuda", lr: float = 0.005, weight_decay: float = 5e-4,
               patience: int = 20, hidden_size: int = 20, num_layers: int = 2, dropout_rate: float = 0.6,
               edge_types: Sequence[str] = DEFAULT_EDGE_TYPES, seed: int = 0) -> HCANModel:
    """Train HCAN with Adam and optional validation-loss early stopping."""
    if train_df["dx_group"].nunique() != 2:
        raise ValueError("HCAN training requires exactly two dx_group classes")
    torch.manual_seed(seed)
    active_device = _device(device)
    features = hcan_feature_columns(train_df)
    classes = np.sort(train_df["dx_group"].unique())
    training = train_df.copy()
    training["_hcan_label"] = pd.Categorical(training["dx_group"], categories=classes).codes
    x, labels, graph = _tensors(training, features, edge_types, active_device)
    model = HCANModel(len(features), hidden_size, num_layers, dropout_rate, edge_types).to(active_device)
    model.feature_cols = list(features)
    model.class_values = classes.tolist()
    model.config["feature_cols"] = list(features)
    model.config["class_values"] = model.class_values
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    validation = None
    if val_df is not None:
        if not set(val_df["dx_group"].unique()).issubset(set(classes)):
            raise ValueError("Validation dx_group values must be present in the training data")
        validation_frame = val_df.copy()
        validation_frame["_hcan_label"] = pd.Categorical(validation_frame["dx_group"], categories=classes).codes
        validation = _tensors(validation_frame, features, edge_types, active_device)
    best_state, best_loss, stalled = None, float("inf"), 0
    loss_history: list[float] = []
    val_loss_history: list[float] = []
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad()
        loss = F.cross_entropy(model(x, graph), labels)
        loss.backward(); optimizer.step()
        monitored_loss = float(loss.detach().cpu())
        loss_history.append(monitored_loss)
        if validation is not None:
            model.eval()
            with torch.no_grad():
                monitored_loss = float(F.cross_entropy(model(validation[0], validation[2]), validation[1]).cpu())
            val_loss_history.append(monitored_loss)
        if monitored_loss < best_loss:
            best_loss, stalled, best_state = monitored_loss, 0, copy.deepcopy(model.state_dict())
        else:
            stalled += 1
            if validation is not None and stalled >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.training_metrics = {"loss": best_loss, "epochs_trained": epoch + 1, "loss_history": loss_history}
    if validation is not None:
        model.training_metrics["val_loss_history"] = val_loss_history
    return model
