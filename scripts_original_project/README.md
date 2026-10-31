# Original project scripts (analysis and figure generation)

> [!CAUTION]
> **These scripts do not run out of the box.** They need the original measurement files, which are not redistributed here, and a path edit:
> 1. Download the datasets (links in the top-level README) and arrange them as
>    `Datasets/Sub_THz_measurements/20cm.mat ... 100cm.mat` and
>    `Datasets/THz_Wireless_Measurements/THz_measurement_channel_frequency_response.mat`.
> 2. Set `BASE_DIR = Path(r"PATH/TO/Datasets")` at the top of each script to that `Datasets` folder.
> 3. `block2_paper_figures.py` also reads `SJTU_Global_Results.mat` (simulation data shared by the SJTU laboratory; not public). Without it the cross-environment figure cannot be regenerated; contact the authors.
> 3b. `round2_analysis.py`, `round2_figures.py` and `leave_two_interior_out.py` have no `BASE_DIR`: they expect `Datasets/` and `block2_output/` in the same folder as the script, and import `block2_complete.py` / `lodo_loao_cv.py` from there.
> 4. Run `block2_complete.py` first (it creates `block2_output/`); the other scripts read from it.

Scripts: `block2_complete.py` (core analysis, models M1-M3, figures 1-12), `block2_paper_figures.py` and `gen_fig_fspl.py` (paper figures), `lodo_loao_cv.py` (LODO/LOAO), `gpr_kernel_sensitivity.py` and `gpr_m5_matern_calibration.py` (GP kernels, M5), `leave_two_interior_out.py`, `round2_analysis.py`, `round2_figures.py` (revision analyses).
For a runnable check without the raw data, use `code/reproduce_validation.py` in the top-level folder.
