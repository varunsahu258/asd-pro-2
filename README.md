# asd_gen_gap

A reproducible scaffold for studying the autism-spectrum-disorder (ASD)
generalization gap. The planned analysis uses internal leave-one-site-out (LOSO)
evaluation on ABIDE I CC200 features, comparing a logistic-regression baseline
with a site-agnostic HCAN (adapted from Shao, Fu & Chen, *BMC Bioinformatics*
2023, with the site meta-path removed for cross-site generalization), followed by one-shot external validation on
ABIDEII-UCLA_1, ABIDEII-KUL_3, ABIDEII-U_MIA_1, and ABIDEII-NYU_2 (32, 28, 28,
and 27 raw subjects, respectively).

ABIDE II is preprocessed with a custom minimal pipeline distinct from ABIDE I's
C-PAC pipeline. The generated limitations text contains the full disclosure of
that cross-pipeline distinction.

## Installation

```bash
pip install -e .
# Optional model and registration dependencies:
pip install -e ".[torch]"
pip install -e ".[ants]"
```

## Project layout

- `data/`: loaders, connectivity construction, minimal preprocessing, and dataset building.
- `models/`: logistic-regression baseline and site-agnostic HCAN implementations.
- `eval/`: LOSO orchestration, statistics, external evaluation, and gap reporting.
- `stats/`: bootstrap confidence intervals, DeLong, McNemar, and Holm--Bonferroni utilities.
- `paper/`: table/figure exporters and limitations-text generation.

## Configuration

Copy or edit `configs/base.yaml` for local paths. `abide_i` and
`abide_ii_raw` are intended as real dataset paths. `yeo_mapping_csv` is the
only initial placeholder and must be set before calling `load_config()`.
