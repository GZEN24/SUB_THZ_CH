# ================================================================
# Script       : lodo_loao_cv.py
# Purpose      : Leave-One-Distance-Out (LODO, 9 folds) and
#                Leave-One-Angle-Out (LOAO, 17 folds) validation.
#                Responds to reviewer comments R2-3, R3-2, and the
#                AE's "generalization analysis" concern.
#
# Reuses your ACTUAL pipeline functions unmodified:
#   fit_separable, sep_predict, fit_poly2d, poly2d_predict,
#   build_mlp, make_2d_features  — all imported from block2_complete
#   GPR kernel — exact match to block2_paper_figures.py Phase 2/3
#
# Place this file in the SAME folder as block2_complete.py:
# <local project folder>
#
# Runtime estimate: under 5 minutes (26 folds total, MLP + GPR
# refit per fold dominate).
# ================================================================

from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from block2_complete import (
    load_all_karakoca, extract_statistics, make_2d_features,
    fit_separable, sep_predict,
    fit_poly2d, poly2d_predict,
    build_mlp,
    ANGLE_CAP_DEG, RANDOM_SEED,
)

np.random.seed(RANDOM_SEED)

# ── DATASET PATH: edit BASE_DIR below ──
BASE_DIR     = Path(r"PATH/TO/Datasets")  # EDIT: folder containing Sub_THz_measurements/ (see README)
KARAKOCA_DIR = BASE_DIR / "Sub_THz_measurements"
OUTPUT_DIR   = SCRIPT_DIR / "block2_output"
OUTPUT_DIR.mkdir(exist_ok=True)

LOO_CV_REF = {"M1": 0.411, "M2": 0.224, "M3": 0.561, "M4": None}
# M4 reference RMSE: paste from your block2_paper_figures.py Table I
# console output once you have it. Left None until confirmed.


# ================================================================
# GPR kernel — EXACT reproduction of block2_paper_figures.py
# ================================================================

def make_gpr_kernel():
    return (
        ConstantKernel(10.0, (1.0, 1000.0))
        * RBF([0.3, 8.0], [(0.01, 5.0), (0.5, 100.0)])
        + WhiteKernel(0.1, (1e-4, 10.0))
    )


def fit_predict_gpr(X_train, y_train, X_test, n_restarts=2):
    gpr = GaussianProcessRegressor(
        kernel=make_gpr_kernel(), alpha=0.0,
        n_restarts_optimizer=n_restarts,
        normalize_y=True, random_state=RANDOM_SEED,
    )
    gpr.fit(X_train, y_train)
    return gpr.predict(X_test)


# ================================================================
# M1 special case — theta = 0 deg held out entirely (LOAO fold 0 only)
# ================================================================

def fit_separable_theta0_held_out(distances_m, angles_deg_train,
                                   pl_matrix_train):
    """
    Fit M1 when the theta=0 (LOS) column is NOT in the training set.
    Only used for the single LOAO fold where theta=0 is withheld.

    Procedure:
      1. Pool all remaining (theta>0) columns to estimate the
         distance slope alpha via per-column log-linear fits,
         averaged across columns.
      2. Per-column intercept c_theta = mean(PL - alpha*log10(d))
         over that column.
      3. Fit c_theta = c0 + a*theta^2 + b*theta over available
         theta>0 values; c0 is the extrapolated LOS intercept beta.
      4. Return w_los=[alpha, c0], w_ang=[a, b] — same format as
         fit_separable, DeltaPL(theta) still zero at theta=0 by
         construction relative to c0.
    """
    log_d = np.log10(distances_m)
    n_ang_train = len(angles_deg_train)

    # Step 1: per-column slope estimates
    slopes = []
    for j in range(n_ang_train):
        A = np.column_stack([log_d, np.ones_like(log_d)])
        w, _, _, _ = np.linalg.lstsq(A, pl_matrix_train[:, j], rcond=None)
        slopes.append(w[0])
    alpha = float(np.mean(slopes))

    # Step 2: per-column intercepts using shared alpha
    intercepts = []
    for j in range(n_ang_train):
        resid = pl_matrix_train[:, j] - alpha * log_d
        intercepts.append(np.mean(resid))
    intercepts = np.array(intercepts)

    # Step 3: quadratic fit of intercepts vs theta -> c0 at theta=0
    theta = angles_deg_train.astype(float)
    A_ang = np.column_stack([theta**2, theta, np.ones_like(theta)])
    coeffs, _, _, _ = np.linalg.lstsq(A_ang, intercepts, rcond=None)
    a_coef, b_coef, c0 = coeffs

    w_los = np.array([alpha, c0])
    w_ang = np.array([a_coef, b_coef])
    return w_los, w_ang


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)))


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    t0 = time.time()

    print("=" * 62)
    print("  PHASE 1 — Loading Karakoca dataset")
    print("=" * 62)
    data_list = load_all_karakoca(KARAKOCA_DIR)
    distances_m, angles_deg, pl_matrix, ds_matrix, bc_matrix = \
        extract_statistics(data_list, angle_cap=ANGLE_CAP_DEG)
    N_dist = len(distances_m)
    N_ang  = len(angles_deg)
    print(f"  Dataset: {N_dist} distances x {N_ang} angles "
          f"= {N_dist*N_ang} points")

    # ─────────────────────────────────────────────────────────
    # LODO — Leave-One-Distance-Out (9 folds)
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  LODO — Leave-One-Distance-Out (9 folds)")
    print("=" * 62)

    lodo_rmse = {"M1": [], "M2": [], "M3": [], "M4": []}

    for k in range(N_dist):
        held_d = distances_m[k]
        train_mask_d = np.ones(N_dist, dtype=bool)
        train_mask_d[k] = False

        d_train = distances_m[train_mask_d]
        pl_train = pl_matrix[train_mask_d, :]      # (8, 17)
        pl_test_row = pl_matrix[k, :]               # (17,) test targets

        print(f"  Fold {k+1}/9: hold out d={held_d*100:.0f} cm "
              f"({N_ang} test pts, {(N_dist-1)*N_ang} train pts)")

        # ---- M1 (theta=0 always present since only a distance removed) ----
        w_los, w_ang = fit_separable(d_train, angles_deg, pl_train)
        y_pred_m1 = sep_predict(held_d, angles_deg, w_los, w_ang)
        lodo_rmse["M1"].append(rmse(pl_test_row, y_pred_m1))

        # ---- M2, M3, M4 (feature-based) ----
        X_train, _, _ = make_2d_features(d_train, angles_deg)
        y_train = pl_train.ravel()
        X_test, _, _  = make_2d_features(np.array([held_d]), angles_deg)
        y_test  = pl_test_row

        w_p2d = fit_poly2d(X_train, y_train)
        y_pred_m2 = poly2d_predict(X_test, w_p2d)
        lodo_rmse["M2"].append(rmse(y_test, y_pred_m2))

        pipe_mlp = build_mlp()
        pipe_mlp.fit(X_train, y_train)
        y_pred_m3 = pipe_mlp.predict(X_test)
        lodo_rmse["M3"].append(rmse(y_test, y_pred_m3))

        y_pred_m4 = fit_predict_gpr(X_train, y_train, X_test)
        lodo_rmse["M4"].append(rmse(y_test, y_pred_m4))

    lodo_avg = {m: float(np.mean(v)) for m, v in lodo_rmse.items()}

    print("\n  LODO per-fold RMSE (dB):")
    print(f"  {'d(cm)':<8} {'M1':>8} {'M2':>8} {'M3':>8} {'M4':>8}")
    for k in range(N_dist):
        print(f"  {distances_m[k]*100:<8.0f} "
              f"{lodo_rmse['M1'][k]:>8.4f} {lodo_rmse['M2'][k]:>8.4f} "
              f"{lodo_rmse['M3'][k]:>8.4f} {lodo_rmse['M4'][k]:>8.4f}")
    print(f"  {'AVG':<8} {lodo_avg['M1']:>8.4f} {lodo_avg['M2']:>8.4f} "
          f"{lodo_avg['M3']:>8.4f} {lodo_avg['M4']:>8.4f}")

    # ─────────────────────────────────────────────────────────
    # LOAO — Leave-One-Angle-Out (17 folds)
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  LOAO — Leave-One-Angle-Out (17 folds)")
    print("=" * 62)

    loao_rmse = {"M1": [], "M2": [], "M3": [], "M4": []}

    for j in range(N_ang):
        held_theta = angles_deg[j]
        train_mask_a = np.ones(N_ang, dtype=bool)
        train_mask_a[j] = False

        a_train = angles_deg[train_mask_a]
        pl_train = pl_matrix[:, train_mask_a]       # (9, 16)
        pl_test_col = pl_matrix[:, j]                # (9,) test targets

        print(f"  Fold {j+1}/17: hold out theta={held_theta:.0f} deg "
              f"({N_dist} test pts, {N_dist*(N_ang-1)} train pts)")

        # ---- M1 ----
        if held_theta == 0.0:
            print("    [special case] theta=0 withheld -> "
                  "using theta0-held-out LOS estimator")
            w_los, w_ang = fit_separable_theta0_held_out(
                distances_m, a_train, pl_train)
        else:
            w_los, w_ang = fit_separable(distances_m, a_train, pl_train)
        y_pred_m1 = sep_predict(distances_m, held_theta, w_los, w_ang)
        loao_rmse["M1"].append(rmse(pl_test_col, y_pred_m1))

        # ---- M2, M3, M4 ----
        X_train, _, _ = make_2d_features(distances_m, a_train)
        y_train = pl_train.ravel()
        X_test, _, _  = make_2d_features(distances_m, np.array([held_theta]))
        y_test  = pl_test_col

        w_p2d = fit_poly2d(X_train, y_train)
        y_pred_m2 = poly2d_predict(X_test, w_p2d)
        loao_rmse["M2"].append(rmse(y_test, y_pred_m2))

        pipe_mlp = build_mlp()
        pipe_mlp.fit(X_train, y_train)
        y_pred_m3 = pipe_mlp.predict(X_test)
        loao_rmse["M3"].append(rmse(y_test, y_pred_m3))

        y_pred_m4 = fit_predict_gpr(X_train, y_train, X_test)
        loao_rmse["M4"].append(rmse(y_test, y_pred_m4))

    loao_avg = {m: float(np.mean(v)) for m, v in loao_rmse.items()}

    print("\n  LOAO per-fold RMSE (dB):")
    print(f"  {'theta':<8} {'M1':>8} {'M2':>8} {'M3':>8} {'M4':>8}")
    for j in range(N_ang):
        print(f"  {angles_deg[j]:<8.0f} "
              f"{loao_rmse['M1'][j]:>8.4f} {loao_rmse['M2'][j]:>8.4f} "
              f"{loao_rmse['M3'][j]:>8.4f} {loao_rmse['M4'][j]:>8.4f}")
    print(f"  {'AVG':<8} {loao_avg['M1']:>8.4f} {loao_avg['M2']:>8.4f} "
          f"{loao_avg['M3']:>8.4f} {loao_avg['M4']:>8.4f}")

    # ─────────────────────────────────────────────────────────
    # SUMMARY + SAVE
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  SUMMARY TABLE (for manuscript Table)")
    print("=" * 62)
    print(f"  {'Protocol':<20} {'M1':>8} {'M2':>8} {'M3':>8} {'M4':>8}")
    print(f"  {'LOO-CV (paper ref)':<20} "
          f"{LOO_CV_REF['M1']:>8.3f} {LOO_CV_REF['M2']:>8.3f} "
          f"{LOO_CV_REF['M3']:>8.3f} {'? (paste M4)':>8}")
    print(f"  {'LODO (avg,9 folds)':<20} "
          f"{lodo_avg['M1']:>8.3f} {lodo_avg['M2']:>8.3f} "
          f"{lodo_avg['M3']:>8.3f} {lodo_avg['M4']:>8.3f}")
    print(f"  {'LOAO (avg,17 folds)':<20} "
          f"{loao_avg['M1']:>8.3f} {loao_avg['M2']:>8.3f} "
          f"{loao_avg['M3']:>8.3f} {loao_avg['M4']:>8.3f}")

    pd.DataFrame({
        "Protocol": ["LODO avg", "LOAO avg"],
        "M1": [lodo_avg["M1"], loao_avg["M1"]],
        "M2": [lodo_avg["M2"], loao_avg["M2"]],
        "M3": [lodo_avg["M3"], loao_avg["M3"]],
        "M4": [lodo_avg["M4"], loao_avg["M4"]],
    }).to_csv(OUTPUT_DIR / "lodo_loao_summary.csv", index=False)

    pd.DataFrame({
        "distance_cm": distances_m * 100,
        "M1": lodo_rmse["M1"], "M2": lodo_rmse["M2"],
        "M3": lodo_rmse["M3"], "M4": lodo_rmse["M4"],
    }).to_csv(OUTPUT_DIR / "lodo_per_fold.csv", index=False)

    pd.DataFrame({
        "angle_deg": angles_deg,
        "M1": loao_rmse["M1"], "M2": loao_rmse["M2"],
        "M3": loao_rmse["M3"], "M4": loao_rmse["M4"],
    }).to_csv(OUTPUT_DIR / "loao_per_fold.csv", index=False)

    print(f"\n  Saved to {OUTPUT_DIR}:")
    print("    lodo_loao_summary.csv")
    print("    lodo_per_fold.csv")
    print("    loao_per_fold.csv")
    print(f"\n  Total runtime: {time.time()-t0:.0f}s")
