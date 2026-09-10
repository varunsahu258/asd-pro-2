"""Build subject-level connectivity feature datasets from ABIDE inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .abide_loader import load_abide_i
from .connectivity import compute_connectivity, flatten_upper_triangle
from .phenotype import filter_by_motion, normalize_phenotypes


_REQUIRED_PHENOTYPE_COLUMNS = {
    "subject_id", "site", "dx_group", "age", "sex", "fiq", "handedness"
}


def _read_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Read only dataset-builder settings, allowing unrelated placeholders."""

    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("dataset_paths"), dict):
        raise ValueError(f"Configuration must define a dataset_paths mapping: {config_path}")
    return config


def _split_sites(sites: str | None, default: list[str] | None = None) -> list[str] | None:
    if sites is None:
        return default
    parsed = [site.strip() for site in sites.split(",") if site.strip()]
    if not parsed:
        raise ValueError("--sites must contain at least one non-empty site name.")
    return parsed


def _load_abide_ii(config: dict[str, Any]) -> pd.DataFrame:
    """Load Prompt 8 processed-array manifest; this is intentionally not ABIDE I loading.

    ABIDE II preprocessing produces CC200 ``.npy`` arrays and a manifest, unlike
    ABIDE I's PCP phenotype CSV plus raw ``.1D`` files.  The explicit branch
    prevents an accidental assumption that both source layouts are interchangeable.
    """

    processed_root = Path(config["dataset_paths"]["abide_ii_processed"])
    if not processed_root.is_dir():
        raise FileNotFoundError(f"ABIDE II processed directory does not exist: {processed_root}")
    manifests = sorted(
        list(processed_root.glob("manifest.csv"))
        + list(processed_root.glob("manifest.parquet"))
    )
    if not manifests:
        raise FileNotFoundError(
            f"No processed ABIDE II manifest (manifest.csv or manifest.parquet) found in {processed_root}"
        )
    manifest_path = manifests[0]
    manifest = pd.read_parquet(manifest_path) if manifest_path.suffix == ".parquet" else pd.read_csv(manifest_path)
    rename_map = {
        "SUB_ID": "subject_id", "SITE_ID": "site", "DX_GROUP": "dx_group",
        "AGE_AT_SCAN": "age", "SEX": "sex", "FIQ": "fiq", "func_mean_fd": "mean_fd",
        "HANDEDNESS_CATEGORY": "handedness",
    }
    manifest = manifest.rename(columns={key: value for key, value in rename_map.items() if key in manifest})
    missing = sorted(_REQUIRED_PHENOTYPE_COLUMNS - set(manifest.columns))
    if missing:
        raise ValueError(f"ABIDE II manifest {manifest_path} is missing required columns: {', '.join(missing)}")
    if "mean_fd" not in manifest:
        raise ValueError(f"ABIDE II manifest {manifest_path} is missing required column: mean_fd")

    manifest = manifest.copy()
    if "handedness" in manifest:
        from .abide_loader import standardize_handedness

        manifest["handedness"] = standardize_handedness(manifest["handedness"])
    manifest["subject_id"] = manifest["subject_id"].astype("string")
    if "timeseries_path" in manifest:
        manifest["timeseries_path"] = manifest["timeseries_path"].map(
            lambda path: str(processed_root / path) if not Path(path).is_absolute() else str(path)
        )
    else:
        paths: list[str] = []
        missing_arrays: list[str] = []
        for subject_id in manifest["subject_id"]:
            matches = sorted(processed_root.rglob(f"{subject_id}_rois_cc200.npy"))
            if not matches:
                missing_arrays.append(str(subject_id))
            else:
                paths.append(str(matches[0]))
        if missing_arrays:
            preview = ", ".join(missing_arrays[:5])
            suffix = "" if len(missing_arrays) <= 5 else ", ..."
            raise FileNotFoundError(
                f"Missing processed ABIDE II CC200 arrays for {len(missing_arrays)} subject(s): {preview}{suffix}"
            )
        manifest["timeseries_path"] = paths
    return manifest.rename(columns={"mean_fd": "func_mean_fd"})


def _load_timeseries(path: str, dataset: str) -> np.ndarray:
    return np.loadtxt(path) if dataset == "I" else np.load(path)


def build_dataset(
    dataset: str, out_path: str | Path, config_path: str | Path, sites: list[str] | None = None,
    max_mean_fd: float | None = None,
) -> pd.DataFrame:
    """Create and persist one parquet row per subject with CC200 connectivity features."""

    dataset = dataset.upper()
    if dataset not in {"I", "II"}:
        raise ValueError("dataset must be 'I' or 'II'.")
    config = _read_yaml_config(config_path)
    source = load_abide_i(config) if dataset == "I" else _load_abide_ii(config)
    selected_sites = sites if sites is not None else (config.get("external_sites") if dataset == "II" else None)
    if selected_sites:
        source = source[source["site"].isin(selected_sites)].copy()
    threshold = max_mean_fd if max_mean_fd is not None else config.get("motion", {}).get("max_mean_fd", 0.2)
    filtered, _ = filter_by_motion(source, float(threshold))
    if selected_sites:
        empty_sites = [site for site in selected_sites if not (filtered["site"] == site).any()]
        if empty_sites:
            raise ValueError(f"Requested site has zero matching subjects: {', '.join(empty_sites)}")

    normalized = normalize_phenotypes(filtered)
    rows: list[dict[str, object]] = []
    for subject in normalized.to_dict(orient="records"):
        connectivity = compute_connectivity(_load_timeseries(subject["timeseries_path"], dataset))
        row = {
            key: subject[key]
            for key in ("subject_id", "site", "dx_group", "age_site_z", "sex", "fiq_site_z", "handedness")
        }
        row["dataset"] = dataset
        row.update(flatten_upper_triangle(connectivity))
        rows.append(row)

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_parquet(output, index=False)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ABIDE CC200 connectivity parquet data.")
    parser.add_argument("--dataset", choices=("I", "II"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sites", help="Comma-separated site IDs.")
    parser.add_argument("--max-mean-fd", type=float, help="Subject-level mean FD exclusion threshold.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = _read_yaml_config(args.config)
    default_sites = config.get("external_sites") if args.dataset == "II" else None
    build_dataset(
        args.dataset, args.out, args.config, _split_sites(args.sites, default_sites), args.max_mean_fd
    )


if __name__ == "__main__":
    main()
