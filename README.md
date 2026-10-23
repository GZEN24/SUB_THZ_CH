# Sub-THz joint distance-angle path-loss models (240-300 GHz)

Code and processed data supporting *"Data-Driven Joint Distance-Angle Path Loss Modeling for Short-Range Sub-THz LoS Links at 240-300 GHz"* (G. Zakka El Nashef, H. Ghanem), IEEE Open Journal of the Communications Society.

**Paper status:** accepted, OJCOMS-07470-2026. `[DOI pending - add at publication]`

The repository implements five path-loss models on the 153-point (9 distances x 17 angles) band-averaged response derived from the open VNA measurements of Karakoca et al. (IEEE DataPort, doi:10.21227/txmh-mz22), and the validation protocols used in the paper: pointwise LOO, leave-one-distance-out (LODO), leave-one-angle-out (LOAO) and a simultaneous 50/80 cm holdout.

| Model | Description |
|---|---|
| M1 | Separable: floating-intercept log-distance + quadratic angular penalty |
| M2 | 2-D polynomial, degree 3, in [log10(d/0.20), theta/20] |
| M3 | MLP (32,16), ReLU, Adam, alpha=0.01, seed 42 |
| M4 | Gaussian process, RBF kernel |
| M5 | Gaussian process, anisotropic radial Matern 5/2 kernel |

## Repository map

```
code/
  subthz_pl.py              models, folds, metrics (self-contained)
  reproduce_validation.py   runs the protocols and checks against the paper's numbers
data/
  pl_targets_153.csv        processed 153-point path-loss table (see data/README.md)
  expected_headline_numbers.csv   values reported in the paper, used by the check
results/
  archived_evidence/        fold predictions, fitted kernels, interval scores, MLP grid,
                            claim ledger from the original runs (frozen)
environment/                pinned requirements
scripts_original_project/   original analysis and figure scripts (need the raw datasets and a path edit)
archive_original_scripts/   run scripts of the final evidence runs, kept for provenance only
docs/REPRODUCIBILITY.md     what reproduces, tolerances, known limits
```

## Quick start

```
pip install -r environment/requirements.txt
python code/reproduce_validation.py            # all protocols, M1-M5 (several minutes)
python code/reproduce_validation.py --skip-loo # grouped protocols only
```

Outputs are written to `results/reproduced/` and a PASS/FAIL table against the paper is printed.

## Raw measurement data (not redistributed)

The original MAT files belong to their authors. Obtain them from IEEE DataPort (access terms are set by each deposit; the DataPort pages indicate an account or subscription may be required):

* Karakoca et al., "sub-THz Wireless Channel Measurements" - https://doi.org/10.21227/txmh-mz22
* Tekbiyik et al., "THz Wireless Channel Measurements in between 240GHz and 300GHz" - https://doi.org/10.21227/2jhd-wp15

The SJTU simulation data used for the contextual comparison are not public. For any of these, or for questions about the data, contact the corresponding author (below).

> [!CAUTION]
> **The scripts in `scripts_original_project/` need the original datasets and a local path edit to work** (see `scripts_original_project/README.md`). `code/reproduce_validation.py` runs without the raw data.

## Not included

The published article PDF and its figures (IEEE copyright; see the article).

## Citation

Cite the article (preferred) and, if used, this archive. `CITATION.cff` provides metadata.

```
[CITATION PENDING - update at publication]
```

## License

Code: MIT (`LICENSE`). Authored result tables and the processed table in `data/` and `results/`: CC BY 4.0, with the data attribution below.
The processed table is derived from the Karakoca et al. dataset; please cite that dataset when using it.

## Contact

Georges Zakka El Nashef, College of Engineering and Technology, American University of the Middle East, Kuwait - georges.el-nashef@aum.edu.kw
