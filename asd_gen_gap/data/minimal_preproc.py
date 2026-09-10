"""Minimal ANTsPy/nilearn preprocessing for the raw ABIDE II layout.

This deliberately uses rigid functional motion correction and **affine-only**
MNI registration. It does not require fMRIPrep, FreeSurfer, or a container
runtime.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn import datasets, image, signal
from nilearn.maskers import NiftiMapsMasker


EXPECTED_SITE_COUNTS = {"ABIDEII-UCLA_1": 32, "ABIDEII-KUL_3": 28, "ABIDEII-U_MIA_1": 28, "ABIDEII-NYU_2": 27}
FD_THRESHOLD_MM = 0.2
HEAD_RADIUS_MM = 50.0


def _ants_module():
    """Import ANTsPy only when preprocessing is invoked, preserving importability."""
    import ants

    return ants


def _as_ants(ants: object, data: np.ndarray, affine: np.ndarray):
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(affine)[:3])
    return ants.from_numpy(np.asarray(data, dtype=np.float32), spacing=spacing)


def _motion_correct(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly align each volume to the middle reference and return 6 parameters."""
    ants = _ants_module()
    reference_index = data.shape[3] // 2
    fixed = _as_ants(ants, data[..., reference_index], affine)
    corrected = np.empty_like(data, dtype=np.float32)
    parameters = np.zeros((data.shape[3], 6), dtype=float)
    for index in range(data.shape[3]):
        if index == reference_index:
            corrected[..., index] = data[..., index]
            continue
        registration = ants.registration(fixed=fixed, moving=_as_ants(ants, data[..., index], affine), type_of_transform="Rigid")
        corrected[..., index] = registration["warpedmovout"].numpy()
        transform = ants.read_transform(registration["fwdtransforms"][0])
        parameters[index] = np.asarray(transform.parameters, dtype=float)[:6]
    return corrected, parameters


def _framewise_displacement(parameters: np.ndarray) -> np.ndarray:
    """Power FD: rotations converted to mm with a 50 mm head radius."""
    derivatives = np.vstack([np.zeros((1, 6)), np.diff(parameters, axis=0)])
    derivatives[:, :3] *= HEAD_RADIUS_MM
    return np.abs(derivatives).sum(axis=1)


def _affine_to_mni(data: np.ndarray, affine: np.ndarray):
    """Apply one affine mean-functional-to-MNI transform to every volume."""
    ants = _ants_module()
    template = datasets.load_mni152_template(resolution=2)
    fixed = ants.image_read(str(template.get_filename()))
    mean_functional = _as_ants(ants, data.mean(axis=3), affine)
    registration = ants.registration(fixed=fixed, moving=mean_functional, type_of_transform="Affine")
    transformed = []
    for index in range(data.shape[3]):
        warped = ants.apply_transforms(fixed=fixed, moving=_as_ants(ants, data[..., index], affine), transformlist=registration["fwdtransforms"])
        transformed.append(warped.numpy())
    output = np.stack(transformed, axis=3).astype(np.float32)
    return nib.Nifti1Image(output, template.affine), "affine_only"


def _tissue_confounds(mni_image: nib.spatialimages.SpatialImage) -> np.ndarray:
    """Extract WM/CSF signals from 0.9-thresholded ICBM152 probability priors."""
    priors = datasets.fetch_icbm152_2009(verbose=0)
    signals = []
    data = mni_image.get_fdata(dtype=np.float32)
    for tissue in (priors.wm, priors.csf):
        prior = image.resample_to_img(tissue, mni_image, interpolation="continuous").get_fdata()
        mask = prior >= 0.9
        if not mask.any():
            raise ValueError("MNI tissue prior has no voxels at the 0.9 threshold")
        signals.append(data[mask, :].mean(axis=0))
    return np.column_stack(signals)


def preprocess_subject(anat_path: str | Path, rest_path: str | Path, out_dir: str | Path) -> dict[str, object]:
    """Preprocess one subject and save cleaned data, motion TSV, and QC JSON.

    Motion regressors, 0.9-thresholded MNI WM/CSF prior signals, and temporal
    0.01--0.1 Hz filtering are applied after affine-only MNI registration.
    """
    started = time.perf_counter()
    anat = nib.load(str(anat_path))  # Validate the required anatomical input.
    if len(anat.shape) != 3:
        raise ValueError("anat_path must be a 3D NIfTI image")
    rest = nib.load(str(rest_path))
    if len(rest.shape) != 4 or rest.shape[3] < 2:
        raise ValueError("rest_path must be a 4D NIfTI image with at least two volumes")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    corrected, motion = _motion_correct(rest.get_fdata(dtype=np.float32), rest.affine)
    fd = _framewise_displacement(motion)
    pd.DataFrame(motion, columns=["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]).to_csv(output / "motion.tsv", sep="\t", index=False)
    mni_image, registration_method = _affine_to_mni(corrected, rest.affine)
    confounds = np.column_stack([motion, _tissue_confounds(mni_image)])
    tr = float(rest.header.get_zooms()[3])
    cleaned_data = signal.clean(mni_image.get_fdata(dtype=np.float32).reshape((-1, rest.shape[3])).T, confounds=confounds, t_r=tr, low_pass=0.1, high_pass=0.01, detrend=True, standardize=None).T.reshape(mni_image.shape)
    cleaned_path = output / "cleaned_rest_mni_affine.nii.gz"
    nib.save(nib.Nifti1Image(cleaned_data.astype(np.float32), mni_image.affine), cleaned_path)
    qc = {"mean_fd": float(fd.mean()), "pct_scrubbed": float((fd > FD_THRESHOLD_MM).mean() * 100), "registration_method": registration_method, "tissue_regression_method": "MNI_prior_threshold", "wall_clock_seconds": time.perf_counter() - started}
    (output / "qc.json").write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
    return {**qc, "cleaned_nii_path": str(cleaned_path), "motion_tsv_path": str(output / "motion.tsv")}


def extract_cc200_timeseries(cleaned_nii_path: str | Path) -> np.ndarray:
    """Extract a timepoints-by-200 CC200 matrix for ``connectivity.py``."""
    atlas = datasets.fetch_atlas_craddock_2012(verbose=0)
    masker = NiftiMapsMasker(maps_img=atlas.scorr_mean, standardize=False)
    values = masker.fit_transform(str(cleaned_nii_path))
    if values.shape[1] != 200:
        raise ValueError(f"Expected CC200 atlas to yield 200 regions, got {values.shape[1]}")
    return values


def _site_lookup(root: Path) -> dict[str, str]:
    """Read an optional ABIDE phenotypic CSV to retain site labels in the manifest."""
    candidates = sorted(path for path in root.glob("*.csv") if "phenotyp" in path.name.lower())
    if not candidates:
        return {}
    phenotypes = pd.read_csv(candidates[0], dtype={"SUB_ID": "string"})
    if not {"SUB_ID", "SITE_ID"}.issubset(phenotypes.columns):
        warnings.warn(f"Phenotype CSV {candidates[0]} lacks SUB_ID/SITE_ID; manifest sites will be Unknown.", RuntimeWarning)
        return {}
    return dict(zip(phenotypes["SUB_ID"].astype(str).str.strip(), phenotypes["SITE_ID"].astype(str), strict=True))


def _subject_rows(root: Path) -> list[tuple[str, str, Path, Path]]:
    sites = _site_lookup(root)
    rows = []
    for subject_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        anat = subject_dir / "session_1" / "anat_1" / "anat.nii.gz"
        rest = subject_dir / "session_1" / "rest_1" / "rest.nii.gz"
        rows.append((subject_dir.name, sites.get(subject_dir.name, "Unknown"), anat, rest))
    return rows


def _process_row(subject_id: str, site: str, anat: Path, rest: Path, out_root: Path) -> dict[str, object]:
    try:
        result = preprocess_subject(anat, rest, out_root / subject_id)
        return {"subject_id": subject_id, "site": site, **{key: result[key] for key in ("mean_fd", "pct_scrubbed", "registration_method", "tissue_regression_method", "wall_clock_seconds")}, "status": "ok", "error_message": ""}
    except Exception as error:
        return {"subject_id": subject_id, "site": site, "mean_fd": np.nan, "pct_scrubbed": np.nan, "registration_method": "affine_only", "tissue_regression_method": "MNI_prior_threshold", "wall_clock_seconds": np.nan, "status": "error", "error_message": str(error)}


def run_batch(root: str | Path, out: str | Path, manifest: str | Path, n_jobs: int = 1) -> pd.DataFrame:
    """Preprocess all layout-matching subjects and append continue-on-error manifest rows."""
    root_path, out_path = Path(root), Path(out)
    rows = _subject_rows(root_path)
    results = Parallel(n_jobs=n_jobs)(delayed(_process_row)(subject, site, anat, rest, out_path) for subject, site, anat, rest in rows)
    report = pd.DataFrame(results)
    manifest_path = Path(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(manifest_path, mode="a", header=not manifest_path.exists(), index=False)
    for site, expected in EXPECTED_SITE_COUNTS.items():
        discovered = int((report["site"] == site).sum())
        if discovered != expected:
            warnings.warn(f"Discovered {discovered} subjects for {site}; expected {expected}.", RuntimeWarning)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Minimal affine-only ABIDE II preprocessing.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    report = run_batch(args.root, args.out, args.manifest, args.n_jobs)
    print(f"Processed {len(report)} subjects: {(report['status'] == 'ok').sum()} succeeded, {(report['status'] == 'error').sum()} failed.")


if __name__ == "__main__":
    main()
