"""Configuration loading with explicit unresolved-placeholder protection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DatasetPaths(BaseModel):
    """Filesystem inputs required to build study datasets."""

    abide_i: str
    abide_ii_raw: str
    abide_ii_processed: str
    yeo_mapping_csv: str


class MotionConfig(BaseModel):
    """Motion-exclusion policy."""

    max_mean_fd: float = Field(ge=0)


class ProjectConfig(BaseModel):
    """Validated project configuration."""

    model_config = ConfigDict(extra="forbid")

    dataset_paths: DatasetPaths
    external_sites: list[str]
    motion: MotionConfig
    random_seed: int
    output_dir: str


def _find_placeholders(value: Any, location: str = "") -> list[str]:
    """Return dotted locations whose string values retain ``PLACEHOLDER``."""

    if isinstance(value, str):
        return [location] if "PLACEHOLDER" in value else []
    if isinstance(value, dict):
        return [
            found
            for key, nested_value in value.items()
            for found in _find_placeholders(nested_value, f"{location}.{key}".strip("."))
        ]
    if isinstance(value, list):
        return [
            found
            for index, nested_value in enumerate(value)
            for found in _find_placeholders(nested_value, f"{location}[{index}]")
        ]
    return []


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load YAML configuration, refusing only unresolved placeholder values.

    Dataset paths are not checked for existence here: ABIDE paths may be mounted
    later or supplied on a separate compute environment.
    """

    config_path = Path(path) if path is not None else Path(__file__).parents[2] / "configs" / "base.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {config_path}")

    unresolved = _find_placeholders(raw_config)
    if unresolved:
        locations = ", ".join(unresolved)
        raise ValueError(
            "Configuration contains unresolved literal 'PLACEHOLDER' value(s) at "
            f"{locations}. Set these value(s) before loading the configuration."
        )

    return ProjectConfig.model_validate(raw_config)
