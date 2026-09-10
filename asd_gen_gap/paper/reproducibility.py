"""Generate a path-safe reproducibility statement."""

from __future__ import annotations

import argparse
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml


def generate_reproducibility_statement(results_dir: str | Path = "results") -> str:
    """Describe repository revision, installed packages, and dataset configuration status."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown (not a git repository)"
    lines = ["# Reproducibility statement", "", f"Git commit: {commit}", "", "## Package versions"]
    for package in ("torch", "torch_geometric", "scikit-learn", "pandas", "numpy"):
        try:
            lines.append(f"- {package}: {version(package)}")
        except PackageNotFoundError:
            continue
    config_path = Path("configs/base.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    lines.extend(["", "## Dataset configuration"])
    for key, value in config.get("dataset_paths", {}).items():
        configured = bool(value) and "PLACEHOLDER" not in str(value)
        lines.append(f"- {key}: {'configured' if configured else 'not configured'}")
    return "\n".join(lines)


def write_reproducibility_statement(output_path: str | Path = "results/reproducibility_statement.md", results_dir: str | Path = "results") -> str:
    text = generate_reproducibility_statement(results_dir)
    destination = Path(output_path); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(text + "\n", encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--results-dir", default="results"); parser.add_argument("--out", default="results/reproducibility_statement.md")
    args = parser.parse_args(argv); write_reproducibility_statement(args.out, args.results_dir)


if __name__ == "__main__":
    main()
