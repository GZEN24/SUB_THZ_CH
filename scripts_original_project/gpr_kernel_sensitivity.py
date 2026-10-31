# ================================================================
# Script       : gpr_kernel_sensitivity.py
# Purpose      : Test whether the theta-length-scale boundary hit
#                (l_theta -> 0.5, the lower bound) materially
#                affects M4-GPR accuracy. Responds to R1-5.
#
# Your ACTUAL M4 kernel (from block2_paper_figures.py) is a SINGLE
# anisotropic kernel object:
#   ConstantKernel(10.0,(1,1000)) * RBF([0.3,8.0],
#       [(0.01,5.0),(0.5,100.0)]) + WhiteKernel(0.1,(1e-4,10.0))
# This anisotropic RBF with length_scale=[l_dist, l_ang] is
# mathematically identical to a product of two independent 1-D
# RBF kernels (one per dimension) — that IS the "product kernel"
# described in the manuscript (Eq. 6). No architecture change
# needed to test the boundary; only the angular bound changes.
#
# Configs tested (all LOO-CV, 153 folds each):
#   Config 0 : baseline, l_ang_min = 0.5   [reproduces block2_paper_figures.py]
#   Config A : extended, l_ang_min = 0.05  [R1-5: extended lower bound]
#   Config B : Matern 5/2 (anisotropic, both axes), l_ang_min = 0.05
#              [R1-5: Matern 5/2 alternative]
#
# Config C (OPTIONAL, advanced, commented out by default):
#   A true mixed kernel — RBF strictly on the distance axis,
#   Matern 5/2 strictly on the angular axis — via the "frozen
#   dimension" trick (fixing one axis's length scale to a huge,
#   non-tunable value in each factor). More faithful to "Matern
#   on the angular axis only" as literally requested by R1, but
#   adds real implementation risk. Included but OFF by default;
#   read the comment block before enabling.
#
# Place this file in the SAME folder as block2_complete.py:
# <local project folder>
#
# Runtime estimate: ~2-4s/fold x 153 folds x 3 configs =
# roughly 15-25 minutes total. Config C (if enabled) adds more.
# ================================================================

from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, ConstantKernel, WhiteKernel
)

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

N_RESTARTS_LOO  = 2   # matches block2_paper_figures.py LOO loop
N_RESTARTS_FULL = 5   # matches block2_paper_figures.py full-data fit


# ================================================================
# KERNEL CONFIGS
# ================================================================

def kernel_config0():
    """Baseline — exact reproduction of block2_paper_figures.py."""
    return (
        ConstantKernel(10.0, (1.0, 1000.0))
        * RBF([0.3, 8.0], [(0.01, 5.0), (0.5, 100.0)])
        + WhiteKernel(0.1, (1e-4, 10.0))
    )


def kernel_configA():
    """Extended lower bound on angular length scale: 0.5 -> 0.05."""
    return (
        ConstantKernel(10.0, (1.0, 1000.0))
        * RBF([0.3, 8.0], [(0.01, 5.0), (0.05, 100.0)])
        + WhiteKernel(0.1, (1e-4, 10.0))
    )


def kernel_configB():
    """Anisotropic Matern 5/2 (both axes), extended angular bound."""
    return (
        ConstantKernel(10.0, (1.0, 1000.0))
        * Matern([0.3, 8.0], [(0.01, 5.0), (0.05, 100.0)], nu=2.5)
        + WhiteKernel(0.1, (1e-4, 10.0))
    )


# ── OPTIONAL Config C — true mixed kernel (RBF-dist x Matern-ang) ──
# Uses the "frozen dimension" trick: each factor kernel is given
# BOTH dimensions, but one dimension's length scale is fixed at a
# huge constant (bounds = (val, val)) so that factor is effectively
# insensitive to that axis. The product of the two factors then
# behaves as a true per-dimension mixed kernel.
# VERIFY behavior before trusting: print kernel_.k1 after fitting
# and confirm the "frozen" entries did not move.
HUGE = 1.0e4

def kernel_configC():
    k_dist_only = RBF([0.3, HUGE], [(0.01, 5.0), (HUGE, HUGE)])
    k_ang_only  = Matern([HUGE, 8.0], [(HUGE, HUGE), (0.05, 100.0)], nu=2.5)
    return (
        ConstantKernel(10.0, (1.0, 1000.0)) * k_dist_only * k_ang_only
        + WhiteKernel(0.1, (1e-4, 10.0))
    )


CONFIGS = [
    ("Config0_baseline", kernel_config0,
     "RBF x RBF, l_ang_min=0.5 (original, reproduces paper Table I)"),
    ("ConfigA_extended", kernel_configA,
     "RBF x RBF, l_ang_min=0.05 (extended lower bound)"),
    ("ConfigB_matern_both", kernel_configB,
     "Matern5/2 x Matern5/2, l_ang_min=0.05"),
    # Uncomment to also run the advanced mixed-kernel config:
    # ("ConfigC_mixed_rbf_matern", kernel_configC,
    #  "RBF(dist-only) x Matern5/2(ang-only), l_ang_min=0.05 [ADVANCED]"),
]


def extract_length_scales(gpr):
    """Pull optimized [l_dist, l_ang] and noise level from fitted kernel."""
    try:
        prod = gpr.kernel_.k1        # ConstantKernel * <anisotropic kernel>
        aniso = prod.k2              # the RBF/Matern factor
        ell = aniso.length_scale     # array [l_dist, l_ang]
        noise = gpr.kernel_.k2.noise_level
        return float(ell[0]), float(ell[1]), float(noise)
    except Exception as e:
        print(f"    [warn] could not extract length scales: {e}")
        return np.nan, np.nan, np.nan


# ================================================================
# MAIN
# ================================================================

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

    results = []

    for config_name, kernel_fn, description in CONFIGS:
        print(f"\n{'='*62}")
        print(f"  {config_name}")
        print(f"  {description}")
        print(f"{'='*62}")
        print(f"  Running LOO-CV ({N} folds, "
              f"{N_RESTARTS_LOO} restarts/fold) ...")

        y_pred = np.zeros(N)
        ell_d_list, ell_a_list = [], []
        t_config = time.time()

        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            gpr = GaussianProcessRegressor(
                kernel=kernel_fn(), alpha=0.0,
                n_restarts_optimizer=N_RESTARTS_LOO,
                normalize_y=True, random_state=RANDOM_SEED,
            )
            gpr.fit(X[mask], y[mask])
            mu = gpr.predict(X[i:i+1])
            y_pred[i] = float(mu[0])

            ell_d, ell_a, _ = extract_length_scales(gpr)
            ell_d_list.append(ell_d)
            ell_a_list.append(ell_a)

            if (i + 1) % 30 == 0 or (i + 1) == N:
                print(f"    fold {i+1}/{N}  "
                      f"({time.time()-t_config:.0f}s elapsed)")

        residuals = y - y_pred
        rmse_val = float(np.sqrt(np.mean(residuals**2)))
        mae_val  = float(np.mean(np.abs(residuals)))
        r2_val   = float(1 - np.sum(residuals**2) /
                          np.sum((y - np.mean(y))**2))
        ell_d_med = float(np.nanmedian(ell_d_list))
        ell_a_med = float(np.nanmedian(ell_a_list))

        print(f"\n  RESULTS — {config_name}")
        print(f"    LOO-CV RMSE : {rmse_val:.4f} dB")
        print(f"    LOO-CV MAE  : {mae_val:.4f} dB")
        print(f"    R2          : {r2_val:.6f}")
        print(f"    l_dist (med): {ell_d_med:.4f}")
        print(f"    l_ang  (med): {ell_a_med:.4f}")
        print(f"    Runtime     : {time.time()-t_config:.0f}s")

        results.append({
            "Config": config_name, "Description": description,
            "LOO_RMSE_dB": rmse_val, "LOO_MAE_dB": mae_val, "R2": r2_val,
            "ell_dist_median": ell_d_med, "ell_ang_median": ell_a_med,
        })

    # ─────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  SUMMARY — all configs")
    print("=" * 62)
    print(f"  {'Config':<24} {'RMSE(dB)':>10} {'R2':>10} "
          f"{'l_dist':>8} {'l_ang':>8}")
    for r in results:
        print(f"  {r['Config']:<24} {r['LOO_RMSE_dB']:>10.4f} "
              f"{r['R2']:>10.6f} {r['ell_dist_median']:>8.4f} "
              f"{r['ell_ang_median']:>8.4f}")

    baseline_rmse = results[0]["LOO_RMSE_dB"]
    print("\n  INTERPRETATION GUIDE:")
    for r in results[1:]:
        delta = r["LOO_RMSE_dB"] - baseline_rmse
        print(f"    {r['Config']}: delta RMSE = {delta:+.4f} dB vs baseline")
        if abs(delta) < 0.01:
            print("      -> boundary hit does NOT materially affect "
                  "accuracy. Use 'near-identical performance' paragraph.")
        elif delta < -0.01:
            print(f"      -> improves by {-delta:.4f} dB. Update Table I "
                  "with new M4 values; note improvement in manuscript.")
        else:
            print(f"      -> {delta:.4f} dB WORSE. Original RBF config "
                  "remains the better choice; report as a negative "
                  "control confirming the original kernel was adequate.")

    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "gpr_kernel_sensitivity_results.csv", index=False)
    print(f"\n  Saved: {OUTPUT_DIR / 'gpr_kernel_sensitivity_results.csv'}")
    print(f"  Total runtime: {(time.time()-t0)/60:.1f} min")
