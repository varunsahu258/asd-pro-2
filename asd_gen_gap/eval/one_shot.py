"""Guards for evaluation artifacts that must not be accidentally replaced."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def assert_one_shot_outputs(outputs: Iterable[str | Path], override: bool = False) -> None:
    """Reject existing output paths unless an explicit override was requested.

    Evaluation predictions are expensive and can be mistaken for a fresh run,
    so callers must opt in before replacing an existing artifact.
    """
    if override:
        return
    existing = [str(Path(output)) for output in outputs if Path(output).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite one-shot output(s): " + ", ".join(existing)
            + ". Pass override=True to replace them."
        )
