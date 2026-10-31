# ================================================================
# Script       : gpr_m5_matern_calibration.py
# Purpose      : M5 (Matern 5/2 GPR) full LOO-CV with predictive
#                std, to compute 90% PI half-width and empirical
#                coverage — matching the manuscript's Section V.F
#                methodology for M4, so M5 is directly comparable.
#
# This is Config B from gpr_kernel_sensitivity.py, rerun with
# return_std=True captured per fold (the sensitivity script only
# tracked RMSE/R²/length scales, not predictive std).
#
# Kernel (unchanged from the run that gave RMSE=0.0776 dB):
#   ConstantKernel(10,(1,1000)) * Matern([0.3,8.0],
#       [(0.01,5.0),(0.05,100.0)], nu=2.5) + WhiteKernel(0.1,(1e-4,10))
#
# Place this file in the SAME folder as block2_complete.py:
# <local project folder>
#
# Runtime estimate: ~75s (same as ConfigB in the sensitivity run).
# ================================================================

from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from block2_complete import (
    load_all_karakoca, extract_statistics, make_2d_features,
    ANGLE_CAP_DEG, RANDOM_SEED,
)

np.random.seed(RANDOM_SEED)

BASE_DIR     = Path(r"PATH/TO/Datasets")  # EDIT: folder containing Sub_THz_measurements/ (see README)
KARAKOCA_DIR = BASE_DIR / "Sub_THz_measurements"
OUTPUT_DIR   = SCRIPT_DIR / "block2_output"
OUTPUT_DIR.mkdir(exist_ok=True)

N_RESTARTS = 2  # matches the sensitivity-run LOO loop


def kernel_m5():
    """M5 kernel — identical to ConfigB in gpr_kernel_sensitivity.py."""
    return (
        ConstantKernel(10.0, (1.0, 1000.0))
        * Matern([0.3, 8.0], [(0.01, 5.0), (0.05, 100.0)], nu=2.5)
        + WhiteKernel(0.1, (1e-4, 10.0))
    )


if __name__ == "__main__":
    t0 = time.time()

    print("=" * 62)
    print("  Loading Karakoca dataset")
    print("=" * 62)
    data_list = load_all_karakoca(KARAKOCA_DIR)
    distances_m, angles_deg, pl_matrix, ds_matrix, bc_matrix = \
        extract_statistics(data_list, angle_cap=ANGLE_CAP_DEG)

    X, D_flat, A_flat = make_2d_features(distances_m, angles_deg)
    y = pl_matrix.ravel()
    N = len(y)
    print(f"  Dataset: {N} points")

    print("\n" + "=" * 62)
    print("  M5 — Matern 5/2 GPR — LOO-CV with predictive std")
    print("=" * 62)

    y_pred = np.zeros(N)
    y_std  = np.zeros(N)
    ell_d_list, ell_a_list = [], []

    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        gpr = GaussianProcessRegressor(
            kernel=kernel_m5(), alpha=0.0,
            n_restarts_optimizer=N_RESTARTS,
            normalize_y=True, random_state=RANDOM_SEED,
        )
        gpr.fit(X[mask], y[mask])
        mu, sg = gpr.predict(X[i:i+1], return_std=True)
        y_pred[i] = float(mu[0])
        y_std[i]  = float(sg[0])

        try:
            aniso = gpr.kernel_.k1.k2
            ell_d_list.append(float(aniso.length_scale[0]))
            ell_a_list.append(float(aniso.length_scale[1]))
        except Exception:
            pass

        if (i + 1) % 30 == 0 or (i + 1) == N:
            print(f"    fold {i+1}/{N}  ({time.time()-t0:.0f}s elapsed)")

    residuals = y - y_pred
    rmse_val  = float(np.sqrt(np.mean(residuals**2)))
    mae_val   = float(np.mean(np.abs(residuals)))
    r2_val    = float(1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2))

    # 90% PI half-width — same convention as manuscript Section V.F:
    # z=1.645 for a two-sided 90% interval, half-width = 1.645*sigma
    half_pi = 1.645 * y_std
    pi90_width = float(np.mean(2.0 * half_pi))          # full-width avg (Table I convention: half-width reported)
    pi90_halfwidth_avg = float(np.mean(half_pi))

    coverage = float(np.mean(np.abs(residuals) <= half_pi))

    ell_d_med = float(np.nanmedian(ell_d_list)) if ell_d_list else float("nan")
    ell_a_med = float(np.nanmedian(ell_a_list)) if ell_a_list else float("nan")

    print("\n" + "=" * 62)
    print("  M5 RESULTS — Matern 5/2 GPR")
    print("=" * 62)
    print(f"  LOO-CV RMSE          : {rmse_val:.4f} dB")
    print(f"  LOO-CV MAE           : {mae_val:.4f} dB")
    print(f"  R2                   : {r2_val:.6f}")
    print(f"  90% PI half-width    : {pi90_halfwidth_avg:.4f} dB  (median: {np.median(half_pi):.4f})")
    print(f"  Empirical coverage   : {coverage*100:.1f}%  (nominal: 90%)")
    print(f"  l_dist (median)      : {ell_d_med:.4f}  (standardised)")
    print(f"  l_ang  (median)      : {ell_a_med:.4f}  (standardised)")
    print(f"  Runtime              : {time.time()-t0:.0f}s")

    print("\n" + "=" * 62)
    print("  MANUSCRIPT TABLE ROW (paste into Section V.C M5 table)")
    print("=" * 62)
    print(f"  M5 Matern5/2 & Product Matern & "
          f"{rmse_val:.3f} & {mae_val:.3f} & {r2_val:.4f} & "
          f"{pi90_halfwidth_avg:.3f} & {coverage*100:.1f}\\% \\\\")

    pd.DataFrame({
        "y_true": y, "y_pred_m5": y_pred, "y_std_m5": y_std,
        "residual": residuals,
    }).to_csv(OUTPUT_DIR / "m5_matern_loo_predictions.csv", index=False)

    print(f"\n  Saved: {OUTPUT_DIR / 'm5_matern_loo_predictions.csv'}")
