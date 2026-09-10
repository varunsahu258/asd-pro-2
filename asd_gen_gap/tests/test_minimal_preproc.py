"""Synthetic tests for the dependency-light ABIDE II preprocessing orchestration."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from asd_gen_gap.data import minimal_preproc


def _write_nifti(path: Path, shape: tuple[int, ...]) -> None:
    image = nib.Nifti1Image(np.random.default_rng(4).normal(size=shape).astype(np.float32), np.eye(4))
    image.header.set_zooms((2.0, 2.0, 2.0, 2.0) if len(shape) == 4 else (2.0, 2.0, 2.0))
    nib.save(image, path)


def test_preprocess_subject_writes_cleaned_outputs_with_synthetic_registration(monkeypatch, tmp_path: Path) -> None:
    anat, rest = tmp_path / "anat.nii.gz", tmp_path / "rest.nii.gz"
    _write_nifti(anat, (3, 3, 3)); _write_nifti(rest, (3, 3, 3, 40))

    monkeypatch.setattr(minimal_preproc, "_motion_correct", lambda data, affine: (data, np.zeros((data.shape[3], 6))))
    monkeypatch.setattr(minimal_preproc, "_affine_to_mni", lambda data, affine: (nib.Nifti1Image(data, affine), "affine_only"))
    monkeypatch.setattr(minimal_preproc, "_tissue_confounds", lambda image: np.zeros((image.shape[3], 2)))

    result = minimal_preproc.preprocess_subject(anat, rest, tmp_path / "output")

    assert nib.load(result["cleaned_nii_path"]).shape == (3, 3, 3, 40)
    assert pd.read_csv(result["motion_tsv_path"], sep="\t").shape == (40, 6)
    assert result["registration_method"] == "affine_only"


def test_cc200_extraction_returns_timepoints_by_200(monkeypatch, tmp_path: Path) -> None:
    cleaned = tmp_path / "cleaned.nii.gz"; _write_nifti(cleaned, (3, 3, 3, 7))

    class FakeMasker:
        def __init__(self, **kwargs):
            assert "maps_img" in kwargs

        def fit_transform(self, path: str) -> np.ndarray:
            return np.zeros((7, 200))

    monkeypatch.setattr(minimal_preproc.datasets, "fetch_atlas_craddock_2012", lambda **kwargs: type("Atlas", (), {"scorr_mean": "atlas"})())
    monkeypatch.setattr(minimal_preproc, "NiftiMapsMasker", FakeMasker)
    assert minimal_preproc.extract_cc200_timeseries(cleaned).shape == (7, 200)


def test_batch_writes_manifest_and_logs_corrupt_subject(monkeypatch, tmp_path: Path) -> None:
    root, output, manifest = tmp_path / "raw", tmp_path / "processed", tmp_path / "manifest.csv"
    for subject in ("good", "corrupt"):
        (root / subject / "session_1" / "anat_1").mkdir(parents=True)
        (root / subject / "session_1" / "rest_1").mkdir(parents=True)
    _write_nifti(root / "good" / "session_1" / "anat_1" / "anat.nii.gz", (3, 3, 3))
    _write_nifti(root / "good" / "session_1" / "rest_1" / "rest.nii.gz", (3, 3, 3, 4))
    (root / "corrupt" / "session_1" / "anat_1" / "anat.nii.gz").write_text("not a nifti")
    (root / "corrupt" / "session_1" / "rest_1" / "rest.nii.gz").write_text("not a nifti")

    def fake_preprocess(anat_path, rest_path, out_dir):
        if Path(anat_path).parent.parent.parent.name == "corrupt":
            raise ValueError("corrupt NIfTI")
        return {"mean_fd": 0.1, "pct_scrubbed": 0.0, "registration_method": "affine_only", "tissue_regression_method": "MNI_prior_threshold", "wall_clock_seconds": 1.2}

    monkeypatch.setattr(minimal_preproc, "preprocess_subject", fake_preprocess)
    report = minimal_preproc.run_batch(root, output, manifest, n_jobs=1)
    persisted = pd.read_csv(manifest)
    assert report["status"].tolist() == ["error", "ok"]
    assert persisted["status"].tolist() == ["error", "ok"]
    assert "corrupt NIfTI" in persisted.loc[0, "error_message"]
