# Sub-THz joint distance-angle path-loss models (240-300 GHz)

Code and processed data supporting *"Data-Driven Joint Distance-Angle Path Loss Modeling for Short-Range Sub-THz LoS Links at 240-300 GHz"* (G. Zakka El Nashef, H. Ghanem), IEEE Open Journal of the Communications Society.

**Published article:** [IEEE Open Journal of the Communications Society (2026)](https://doi.org/10.1109/OJCOMS.2026.3747701), DOI: **[10.1109/OJCOMS.2026.3747701](https://doi.org/10.1109/OJCOMS.2026.3747701)**. Submission reference: OJCOMS-07470-2026.

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

Python 3.10 or 3.11 is recommended for the pinned reference dependencies. From a terminal at the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r environment/requirements.txt
python -m unittest discover -s tests -v
python code/reproduce_validation.py --skip-loo --skip-mlp  # fast 26-check verification
python code/reproduce_validation.py                          # all 36 checks, slow
```

On Windows, `py -3.11` may replace `python` when the Python launcher is installed. If `py -3.10` is unavailable, install a supported Python version rather than relying on the system Python.

Output is written to `results/reproduced/`. The script returns a nonzero exit status for missing or failed selected paper checks. Flags `--skip-loo`, `--skip-mlp` and `--skip-grouped` select subsets; such runs must **not** be reported as complete reproduction. GitHub Actions runs the fast subset on every push/PR, while its manual **Run workflow** option supports the full reproduction.

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

```bibtex
@article{ZakkaElNashef2026SubTHzPathLoss,
  author  = {Georges Zakka El Nashef and Haitham Ghanem},
  title   = {Data-Driven Joint Distance-Angle Path Loss Modeling for Short-Range Sub-THz LoS Links at 240-300 GHz},
  journal = {IEEE Open Journal of the Communications Society},
  year    = {2026},
  doi     = {10.1109/OJCOMS.2026.3747701}
}
```

The paper DOI is not a DOI for this GitHub repository. Add an archive DOI to `CITATION.cff` **only after** a Zenodo release has minted one.

## License

Code: MIT (`LICENSE`). The authors intend CC BY 4.0 for original result tables and their own contributions to the processed data. **This does not license the underlying third-party measurements.** The source DataPort license/permission for redistributing derived scalar values has not been independently established; verify those terms before making the processed table public. Cite Karakoca et al. when using the derived data. See `data/README.md`.

## Contact

Georges Zakka El Nashef, College of Engineering and Technology, American University of the Middle East, Kuwait - georges.el-nashef@aum.edu.kw
