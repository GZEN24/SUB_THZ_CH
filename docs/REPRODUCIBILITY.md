# Reproducibility notes

## What reproduces
`python code/reproduce_validation.py` re-fits M1-M5 from `data/pl_targets_153.csv` and recomputes:
pointwise LOO RMSE (M1-M5), LODO and LOAO (mean-fold and pooled RMSE), the simultaneous 50/80 cm holdout, and the grouped 90 % interval coverage of M4/M5.
`results/reproduced/comparison_with_paper.csv` (written by the script) lists 36 values against the paper.

Check run (2026-10-24, Python 3.13, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.5, Linux):
**36 of 36 cells within tolerance**; largest RMSE difference 0.0005 dB (the paper reports three decimals), largest coverage difference 0.004 percentage points.
The original results were produced with scikit-learn 1.3.2 (`environment/requirements.txt`); the match on the newer stack indicates the numbers are not sensitive to the library version at the reported precision.

## Conventions (as in the paper)
* Features `[log10(d/0.20), theta/20]`; no scaler for M1, M2, M4, M5; MLP uses a training-fold `StandardScaler`.
* GP: `C(10,(1,1000)) * kernel + White(0.1,(1e-4,10))`, `normalize_y=True`, `alpha=0`, L-BFGS-B, `random_state=42`, two restarts plus the initial optimization per fit.
  M4 length-scale bounds [(0.01,5),(0.5,100)]; M5 [(0.01,5),(0.05,100)].
* 90 % intervals: `z = 1.645` with the WhiteKernel-inclusive observation standard deviation; interval score = width + 20 x miss.
* M1 uses a dedicated fit for the single LOAO fold that withholds theta = 0 (`fit_separable_no_los`).

## What is archived rather than re-runnable
`results/archived_evidence/` holds the frozen outputs of the original runs (fold-level predictions, fitted kernels, interval scores, the 30-configuration / 810-fit MLP sensitivity grid, the numerical claim ledger).
`archive_original_scripts/` holds the original run scripts for provenance. They depend on the authors' local project modules and the raw MAT files and are not runnable on their own; local absolute paths were removed.
The MLP sensitivity grid is archived, not re-run by `reproduce_validation.py`.

## Not included
Raw third-party measurements and the SJTU simulation data (links and contact in the README), and the article PDF. The figure and analysis scripts are in `scripts_original_project/` and need the raw data.

## Release validation and automated checks (2026-10-31)
The automated push/PR workflow checks the 26 grouped non-MLP metrics and runs unit tests. The full 36-cell reproduction must be triggered manually in GitHub Actions or run locally; the default command is computationally expensive. A subset passing does not certify the full 36-cell set. The reproduction command exits nonzero on any failed or missing selected comparison, and malformed/reordered target grids are rejected before fitting. No fitted scientific model was changed by these release guardrails.
