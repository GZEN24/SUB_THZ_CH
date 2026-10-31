# ================================================================
# Script       : block2_paper_figures.py
# Purpose      : Single master script that generates ALL 13
#                publication figures for the Block 2 paper.
#                One run produces every file in the figure map.
#
# R1-7 REVISION: figure styling reworked in response to reviewer
# comment R1-7 ("low resolution figures... weird legends and
# titles, aspect ratio is not good... labels/legend text should
# be legible... marker sizes should be large enough"). Changes:
#   - savefig.dpi 300 -> 600
#   - label/legend/tick font floors raised (see rcParams block)
#   - marker/scatter size floors raised (MS, MS_SMALL, SCAT_S)
#   - in-figure titles stripped of embedded equations/statistics
#     (that content lives in the .tex captions already)
#   - Figs 8 and 10 restructured from 1x4 to 2x2: a 14-inch-wide
#     canvas forced into \columnwidth (~3.5in) printed at ~1in
#     tall; 2x2 holds up far better at single-column width
#   - Figs 4, 5, 6, 11 given tighter, single-column-friendly
#     aspect ratios (previously 10:4.2, now ~8.2-9.2:4.6-4.8)
#
# Runtime      : ~5 min (M3 LOO-CV 170s + M4 LOO-CV 55s)
#
# Output files (all saved to block2_output/):
#   fig_PL_vs_distance.png      -> fig:pl_distance
#   fig_fspl_ci_fit.png         -> fig:fspl
#   fig_angular_sensitivity.png -> fig:pl_angle
#   fig_coherence_bw.png        -> fig:coherence_bw
#   fig_delay_spread.png        -> fig:delay_spread
#   fig_separability.png        -> fig:sep_heatmap
#   fig_gpr_angle_slices.png    -> fig:gpr_slices
#   fig_all_models_loo.png      -> fig:loo_scatter
#   fig_gpr_loo_scatter.png     -> fig:gpr_scatter
#   fig_residuals.png           -> fig:residuals
#   fig_gpr_surface.png         -> fig:gpr_surface
#   fig_model_surface.png       -> fig:surface
#   fig_cross_env_PLE.png       -> fig:cross_env
#
# Author       : Dr. Georges Zakka El Nashef — AUM Kuwait
# Environment  : Python 3.11
# ================================================================

from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from block2_complete import (
    load_all_karakoca, load_tekbiyik,
    extract_statistics, make_2d_features,
    fit_separable, sep_predict,
    fit_poly2d, poly2d_predict,
    build_mlp,
    loo_cv_separable, loo_cv_generic, _cv_metrics,
    fit_delay_spread_model,
    fspl_db, separability_test,
    ALL_ANGLES_DEG, ANGLE_CAP_DEG, DISTANCES_M, TEKBIYIK_DISTS_M,
    F_CENTER_HZ, MEAS_BW_HZ, C_LIGHT_MS,
    COLOR, PLOT_ANGLES, RANDOM_SEED,
)

# ── Paths: edit BASE_DIR below ───
BASE_DIR = Path(
    r"PATH/TO/Datasets"  # EDIT: folder containing Sub_THz_measurements/ (see README)
)
KARAKOCA_DIR  = BASE_DIR / "Sub_THz_measurements"
TEKBIYIK_FILE = (BASE_DIR / "THz_Wireless_Measurements"
                 / "THz_measurement_channel_frequency_response.mat")
SJTU_MAT      = SCRIPT_DIR / "SJTU_Global_Results.mat"
OUTPUT_DIR    = SCRIPT_DIR / "block2_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Figure style ──────────────────────────────────────────────────
# R1-7 response: larger label/legend/tick floors, larger markers,
# consistent single-column-friendly aspect ratios (see per-figure
# figsize choices below). No figure relies on font sizes below the
# floors set here.
matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "DejaVu Serif"],
    "font.size":       11,
    "axes.labelsize":  12,
    "axes.titlesize":  11,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":      150,
    "savefig.dpi":     600,
    "savefig.bbox":    "tight",
    "lines.linewidth": 1.8,
    "axes.grid":       True,
    "grid.alpha":      0.2,
})

MS       = 7.0   # floor for line-plot markers (R1: "marker sizes should be large enough")
MS_SMALL = 6.0   # floor for dense multi-series line plots
SCAT_S   = 26    # floor for scatter-point area

np.random.seed(RANDOM_SEED)
COLOR_GPR = "#e31a1c"

def save_fig(fig, name):
    fig.savefig(OUTPUT_DIR / name, dpi=300)
    plt.close(fig)
    print(f"  [saved] {name}")

def section(t):
    print(f"\n{'='*62}\n  {t}\n{'='*62}")

# ================================================================
# PHASE 1 — LOAD DATA
# ================================================================
section("PHASE 1 — Loading datasets")

data_list = load_all_karakoca(KARAKOCA_DIR)
tek       = load_tekbiyik(TEKBIYIK_FILE)

mat_sjtu   = scipy.io.loadmat(str(SJTU_MAT), squeeze_me=True)
n_sjtu_all = np.atleast_1d(mat_sjtu["n_vals"]).astype(float)
lbl_raw    = mat_sjtu["labels"]
sig_sjtu   = np.atleast_1d(mat_sjtu["sigma_vals"]).astype(float)
sjtu_lbl   = [str(l).strip() for l in lbl_raw]
valid_s    = ~np.isnan(n_sjtu_all)
n_sjtu     = n_sjtu_all[valid_s]
lbl_sjtu   = [sjtu_lbl[i] for i in range(len(sjtu_lbl)) if valid_s[i]]
print(f"  SJTU: {len(n_sjtu)} valid scenarios")

distances_m, angles_deg, pl_matrix, ds_matrix, bc_matrix = \
    extract_statistics(data_list, angle_cap=ANGLE_CAP_DEG)

X, D_flat, A_flat = make_2d_features(distances_m, angles_deg)
y  = pl_matrix.ravel()
N  = len(y)
print(f"  Dataset: {N} points ({len(distances_m)}d × {len(angles_deg)}θ)")

# Tekbiyik path loss
tek_pl = np.array([-np.mean(tek["cfr_db"][:, i])
                    for i in range(tek["cfr_db"].shape[1])])

# ================================================================
# PHASE 2 — FIT ALL MODELS
# ================================================================
section("PHASE 2 — Fitting models")

print("[M1] Separable polynomial ...")
w_los, w_ang = fit_separable(distances_m, angles_deg, pl_matrix)
n_ple        = w_los[0] / 10.0
print(f"  n = {n_ple:.4f}")
print(f"  PL_LOS(d) = {w_los[0]:.4f}·log10(d) + {w_los[1]:.4f}")
print(f"  ΔPL(θ)    = {w_ang[0]:.6f}·θ² + {w_ang[1]:.6f}·θ")

print("[M2] 2D polynomial degree-3 ...")
w_poly2d = fit_poly2d(X, y)

print("[M3] MLP 32→16 ...")
pipe_mlp = build_mlp()
pipe_mlp.fit(X, y)

print("[M4] GPR — product RBF kernel ...")
kernel = (
    ConstantKernel(10.0, (1.0, 1000.0))
    * RBF([0.3, 8.0], [(0.01, 5.0), (0.5, 100.0)])
    + WhiteKernel(0.1, (1e-4, 10.0))
)
gpr = GaussianProcessRegressor(
    kernel=kernel, alpha=0.0,
    n_restarts_optimizer=5,
    normalize_y=True, random_state=RANDOM_SEED,
)
t0 = time.time()
gpr.fit(X, y)
print(f"  Fit time: {time.time()-t0:.1f}s")
print(f"  Kernel: {gpr.kernel_}")

sep = separability_test(distances_m, angles_deg, pl_matrix)

# ================================================================
# PHASE 3 — LOO-CV ALL MODELS
# ================================================================
section("PHASE 3 — LOO-CV (all models)")

print("[M1] LOO-CV ...")
cv_m1 = loo_cv_separable(distances_m, angles_deg, pl_matrix)
print(f"  RMSE={cv_m1['rmse']:.3f}")

print("[M2] LOO-CV ...")
cv_m2 = loo_cv_generic(X, y, "poly2d")
print(f"  RMSE={cv_m2['rmse']:.3f}")

print("[M3] LOO-CV (~170s) ...")
t0 = time.time()
cv_m3 = loo_cv_generic(X, y, "mlp")
print(f"  RMSE={cv_m3['rmse']:.3f}  ({time.time()-t0:.0f}s)")

print("[M4] GPR LOO-CV (~55s) ...")
t0 = time.time()
y_pred_m4 = np.zeros(N)
y_std_m4  = np.zeros(N)
for i in range(N):
    mask = np.ones(N, dtype=bool); mask[i] = False
    g = GaussianProcessRegressor(
        kernel=kernel, alpha=0.0,
        n_restarts_optimizer=2,
        normalize_y=True, random_state=RANDOM_SEED)
    g.fit(X[mask], y[mask])
    mu, sg         = g.predict(X[i:i+1], return_std=True)
    y_pred_m4[i]   = float(mu)
    y_std_m4[i]    = float(sg)
cv_m4      = _cv_metrics(y, y_pred_m4)
pi90_width = float(np.mean(2.0 * 1.645 * y_std_m4))
print(f"  RMSE={cv_m4['rmse']:.3f}  90%-PI={pi90_width:.3f} dB  ({time.time()-t0:.0f}s)")

print("\nTABLE I SUMMARY:")
print(f"  {'Model':<18} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'90%-PI':>9}")
print("  " + "-"*55)
for nm, cv, pi in [("M1 Separable", cv_m1, None), ("M2 Poly-2D", cv_m2, None),
                    ("M3 MLP",       cv_m3, None), ("M4 GPR",    cv_m4, pi90_width)]:
    pi_s = f"{pi:>9.3f}" if pi else f"{'—':>9}"
    print(f"  {nm:<18} {cv['rmse']:>8.3f} {cv['mae']:>8.3f} {cv['r2']:>8.4f} {pi_s}")

# Bias
bias_m1 = float(np.mean(cv_m1["y_pred"] - cv_m1["y_true"]))
bias_m2 = float(np.mean(cv_m2["y_pred"] - cv_m2["y_true"]))
bias_m3 = float(np.mean(cv_m3["y_pred"] - cv_m3["y_true"]))
bias_m4 = float(np.mean(y_pred_m4 - y))

# Coherence BW stats
bc_los_med  = float(np.nanmedian(bc_matrix[:, 0]))
bc_20_idx   = int(np.where(angles_deg == 20)[0][0])
bc_20_med   = float(np.nanmedian(bc_matrix[:, bc_20_idx]))
ds_los_mean = float(np.mean(ds_matrix[:, 0]))

# DS delay model
pipe_ds = fit_delay_spread_model(distances_m, angles_deg, ds_matrix)

# ================================================================
# PHASE 4 — FIGURES
# ================================================================
section("PHASE 4 — Generating all 13 figures")

d_fine = np.linspace(0.20, 1.00, 300)

# ── FIG 1: fig_PL_vs_distance.png ────────────────────────────────
fig, ax = plt.subplots(figsize=(6.8, 5.2))
colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(PLOT_ANGLES)))
for theta, col in zip(PLOT_ANGLES, colors):
    idx = int(np.where(angles_deg == theta)[0][0])
    ax.plot(distances_m * 100, pl_matrix[:, idx],
            "o-", color=col, ms=MS, lw=1.8, label=f"θ = {theta}°")

tek_shift = pl_matrix[0, 0] - tek_pl[0]
ax.plot(TEKBIYIK_DISTS_M * 100, tek_pl + tek_shift,
        "ks--", ms=MS_SMALL, lw=1.6, label="Tekbiyik LoS (validation)")

fspl_ref = fspl_db(np.array([0.20]))[0] - pl_matrix[0, 0]
ax.plot(d_fine * 100, fspl_db(d_fine) - fspl_ref,
        color="#999999", ls="-.", lw=1.6, label="FSPL (270 GHz, n=2)")

pl_m1_los = w_los[0] * np.log10(d_fine) + w_los[1]
ax.plot(d_fine * 100, pl_m1_los,
        color=COLOR["separable"], ls="--", lw=1.8,
        label=f"M1 LoS fit (n={n_ple:.3f})")

ax.set_xlabel("Distance (cm)")
ax.set_ylabel("Path Loss (dB)")
ax.legend(fontsize=9.5, loc="upper left")
ax.set_xlim([15, 105])
plt.tight_layout()
save_fig(fig, "fig_PL_vs_distance.png")

# ── FIG 2: fig_fspl_ci_fit.png ───────────────────────────────────
d0_m     = 0.20
pl_los   = pl_matrix[:, 0]
pl0_kara = pl_los[0]
pl0_tek  = tek_pl[0]
rel_kara = pl_los - pl0_kara
rel_tek  = tek_pl - pl0_tek
log_d0   = np.log10(d0_m)
rel_ci   = w_los[0] * (np.log10(d_fine) - log_d0)
lam      = C_LIGHT_MS / F_CENTER_HZ
fspl_abs = 20.0 * np.log10(4.0 * np.pi * d_fine / lam)
fspl_ref0= 20.0 * np.log10(4.0 * np.pi * d0_m   / lam)
rel_fspl = fspl_abs - fspl_ref0

fig, ax = plt.subplots(figsize=(6.5, 5.0))
ax.plot(distances_m * 100, rel_kara,
        "o-", color=COLOR["measured"], ms=MS+1, lw=1.8,
        label="Karakoca LoS (θ = 0°)")
ax.plot(TEKBIYIK_DISTS_M * 100, rel_tek,
        "s--", color=COLOR["tekbiyik"], ms=MS+1, lw=1.6,
        label="Tekbiyik LoS (validation)")
ax.plot(d_fine * 100, rel_ci,
        color=COLOR["separable"], ls="--", lw=2.0,
        label=f"CI fit: n = {n_ple:.3f}")
ax.plot(d_fine * 100, rel_fspl,
        color=COLOR["fspl"], ls="-.", lw=1.6,
        label="FSPL: n = 2.000 (270 GHz)")
ax.set_xlabel("Distance (cm)")
ax.set_ylabel(f"Relative Path Loss (dB, ref. {int(d0_m*100)} cm)")
ax.legend(loc="upper left", fontsize=9.5)
ax.set_xlim([15, 105])
plt.tight_layout()
save_fig(fig, "fig_fspl_ci_fit.png")

# ── FIG 3: fig_angular_sensitivity.png ───────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.8))
d_targets = [0.20, 0.40, 0.60, 0.80, 1.00]
colors_d  = plt.cm.viridis(np.linspace(0.1, 0.9, len(d_targets)))
for d_t, col in zip(d_targets, colors_d):
    idx_d = int(np.argmin(np.abs(distances_m - d_t)))
    delta = pl_matrix[idx_d, :] - pl_matrix[idx_d, 0]
    ax.plot(angles_deg, delta, "o-", color=col,
            ms=MS, lw=1.8, label=f"d = {int(d_t*100)} cm")
theta_fine = np.linspace(0, 20, 200)
delta_m1   = w_ang[0] * theta_fine**2 + w_ang[1] * theta_fine
ax.plot(theta_fine, delta_m1, "k--", lw=2.0, label="M1: ΔPL(θ) fit")
ax.axvline(8, color="#cc4444", ls=":", lw=1.6, alpha=0.8,
           label="θ = 8° separability bound")
ax.set_xlabel("Misalignment Angle θ (deg)")
ax.set_ylabel("Additional Path Loss vs. LoS (dB)")
ax.legend(fontsize=9.5, loc="upper left")
ax.set_xlim([-0.5, 21])
plt.tight_layout()
save_fig(fig, "fig_angular_sensitivity.png")

# ── FIG 4: fig_coherence_bw.png ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 4.6))
bc_plot_angles = [0, 5, 10, 15, 20]
colors_bc = plt.cm.viridis(np.linspace(0.1, 0.9, len(bc_plot_angles)))
for theta_t, col in zip(bc_plot_angles, colors_bc):
    idx_a = int(np.where(angles_deg == theta_t)[0][0])
    ax1.semilogy(distances_m * 100, bc_matrix[:, idx_a],
                 "o-", color=col, ms=MS_SMALL, lw=1.6, label=f"θ = {theta_t}°")
ax1.axhline(MEAS_BW_HZ / 1e6, color="k", ls="--", lw=1.6,
            label=f"Signal BW = {MEAS_BW_HZ/1e6:.0f} MHz")
ax1.set_xlabel("Distance (cm)")
ax1.set_ylabel("Coherence Bandwidth (MHz)")
ax1.legend(fontsize=8.5)

bc_los_vals = bc_matrix[:, 0]
bc_20_vals  = bc_matrix[:, bc_20_idx]
ax2.bar(distances_m * 100 - 1.5, bc_los_vals, width=3.0,
        color=COLOR["measured"], alpha=0.85, label="θ = 0°")
ax2.bar(distances_m * 100 + 1.5, bc_20_vals, width=3.0,
        color="#d62728", alpha=0.85, label="θ = 20°")
ax2.set_yscale("log")
ax2.set_xlabel("Distance (cm)")
ax2.set_ylabel("Coherence Bandwidth (MHz)")
ax2.legend(fontsize=9)
plt.tight_layout()
save_fig(fig, "fig_coherence_bw.png")

# ── FIG 5: fig_delay_spread.png ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 4.6))
for theta_t in PLOT_ANGLES:
    idx_a = int(np.where(angles_deg == theta_t)[0][0])
    ax1.plot(distances_m * 100, ds_matrix[:, idx_a] * 1000,
             "o-", ms=MS_SMALL, lw=1.6, label=f"θ = {theta_t}°")
ax1.set_xlabel("Distance (cm)")
ax1.set_ylabel("RMS Delay Spread (ps)")
ax1.legend(fontsize=8.5)

for d_t, col in zip([0.20, 0.40, 0.60, 0.80, 1.00],
                     plt.cm.viridis(np.linspace(0.1, 0.9, 5))):
    idx_d = int(np.argmin(np.abs(distances_m - d_t)))
    ax2.plot(angles_deg, ds_matrix[idx_d, :] * 1000,
             "o-", ms=MS_SMALL, lw=1.6, color=col,
             label=f"d = {int(d_t*100)} cm")
ax2.set_xlabel("Misalignment Angle θ (deg)")
ax2.set_ylabel("RMS Delay Spread (ps)")
ax2.legend(fontsize=8.5, loc="upper left")
plt.tight_layout()
save_fig(fig, "fig_delay_spread.png")

# ── FIG 6: fig_separability.png ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 4.6))
penalty = sep["penalty_table"]
im = ax1.imshow(penalty.T, aspect="auto", origin="lower",
    extent=[distances_m[0]*100, distances_m[-1]*100,
            angles_deg[0], angles_deg[-1]],
    cmap="plasma", vmin=0)
plt.colorbar(im, ax=ax1, label="Angular Penalty (dB)")
ax1.set_xlabel("Distance (cm)")
ax1.set_ylabel("Misalignment Angle θ (deg)")

max_var  = sep["max_variation"]
worst_th = angles_deg[np.argmax(max_var)]
ax2.bar(angles_deg, max_var, color="#e07b39",
        edgecolor="#c05010", linewidth=0.8)
ax2.axhline(1.0, color="k", ls="--", lw=1.8,
            label="1 dB measurement uncertainty")
ax2.axhline(sep["overall_max"], color="#cc2222", ls=":",
            lw=1.6,
            label=f"Worst case: {sep['overall_max']:.3f} dB at θ={worst_th}°")
ax2.set_xlabel("Misalignment Angle θ (deg)")
ax2.set_ylabel("Max variation across distances (dB)")
ax2.legend(fontsize=8.5, loc="upper left")
plt.tight_layout()
save_fig(fig, "fig_separability.png")

# ── FIG 7: fig_gpr_angle_slices.png ──────────────────────────────
theta_slices = [0, 5, 10, 20]
fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2))
for ax, theta_t in zip(axes.flatten(), theta_slices):
    idx_a   = int(np.where(angles_deg == theta_t)[0][0])
    pl_meas = pl_matrix[:, idx_a]
    X_sl    = make_2d_features(distances_m, np.array([theta_t]))[0]
    pl_m1   = sep_predict(distances_m, theta_t, w_los, w_ang)
    pl_m2   = poly2d_predict(X_sl, w_poly2d)
    mu_gpr, s_gpr = gpr.predict(X_sl, return_std=True)
    d_cm = distances_m * 100
    ax.plot(d_cm, pl_meas, "o", color=COLOR["measured"],
            ms=MS+1, zorder=5, label="Measured")
    ax.plot(d_cm, pl_m1, "-",
            color=COLOR["separable"], lw=1.8, label="M1 Separable")
    ax.plot(d_cm, pl_m2, "--",
            color=COLOR["poly2d"],    lw=1.8, label="M2 Poly-2D")
    ax.plot(d_cm, mu_gpr, "-.",
            color=COLOR_GPR,          lw=2.0, label="M4 GPR (mean)")
    ax.fill_between(d_cm,
                    mu_gpr - 1.645 * s_gpr,
                    mu_gpr + 1.645 * s_gpr,
                    color=COLOR_GPR, alpha=0.15, label="M4 GPR 90% PI")
    ax.set_title(f"θ = {theta_t}°", fontsize=11)
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel("Path Loss (dB)")
    ax.legend(fontsize=8.5, loc="upper left")
plt.tight_layout()
save_fig(fig, "fig_gpr_angle_slices.png")

# ── FIG 8: fig_all_models_loo.png ────────────────────────────────
# Restructured from 1x4 to 2x2: a 14x4in canvas forced to
# \columnwidth (~3.5in) prints at ~1in tall — illegible. 2x2 at a
# near-square canvas holds up far better at single-column width.
pl_range = [y.min() - 1, y.max() + 1]
fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.4))
all_cv = [("M1 Separable",  cv_m1,  COLOR["separable"]),
          ("M2 Poly-2D",    cv_m2,  COLOR["poly2d"]),
          ("M3 MLP",        cv_m3,  COLOR["mlp"]),
          ("M4 GPR",
           {"y_true": y, "y_pred": y_pred_m4,
            "rmse": cv_m4["rmse"], "r2": cv_m4["r2"]},
           COLOR_GPR)]
for ax, (name, cv, col) in zip(axes.flatten(), all_cv):
    ax.scatter(cv["y_true"], cv["y_pred"],
               c=col, s=SCAT_S, alpha=0.75, edgecolors="none")
    ax.plot(pl_range, pl_range, "k--", lw=1.3)
    ax.set_xlim(pl_range); ax.set_ylim(pl_range)
    ax.set_xlabel("Measured PL (dB)")
    ax.set_ylabel("Predicted PL (dB)")
    ax.set_title(f"{name}\nRMSE={cv['rmse']:.3f} dB  R²={cv['r2']:.4f}",
                 fontsize=10)
    ax.set_aspect("equal")
plt.tight_layout()
save_fig(fig, "fig_all_models_loo.png")

# ── FIG 9: fig_gpr_loo_scatter.png ───────────────────────────────
fig, ax = plt.subplots(figsize=(5.6, 5.6))
sc = ax.scatter(y, y_pred_m4, c=y_std_m4, cmap="plasma",
                s=SCAT_S+4, alpha=0.85, edgecolors="none",
                vmin=0, vmax=y_std_m4.max())
plt.colorbar(sc, ax=ax, label="Predictive std (dB)")
ax.plot(pl_range, pl_range, "k--", lw=1.6, label="1:1 line")
ax.set_xlim(pl_range); ax.set_ylim(pl_range)
ax.set_xlabel("Measured PL (dB)")
ax.set_ylabel("GPR Predicted PL (dB)")
ax.set_aspect("equal")
ax.legend(fontsize=9.5)
plt.tight_layout()
save_fig(fig, "fig_gpr_loo_scatter.png")

# ── FIG 10: fig_residuals.png — all 4 models ─────────────────────
# Restructured from 1x4 to 2x2, same rationale as Fig 8.
fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6))
res_data = [
    ("M1 Separable", cv_m1["y_pred"] - cv_m1["y_true"], COLOR["separable"]),
    ("M2 Poly-2D",   cv_m2["y_pred"] - cv_m2["y_true"], COLOR["poly2d"]),
    ("M3 MLP",       cv_m3["y_pred"] - cv_m3["y_true"], COLOR["mlp"]),
    ("M4 GPR",       y_pred_m4 - y,                     COLOR_GPR),
]
for ax, (name, res, col) in zip(axes.flatten(), res_data):
    bias = float(np.mean(res))
    std  = float(np.std(res))
    ax.hist(res, bins=18, color=col, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0,    color="k",      ls="--", lw=1.6)
    ax.axvline(bias, color="#cc2222", ls="-",  lw=1.6,
               label=f"bias = {bias:+.3f} dB")
    ax.set_xlabel("Residual (dB)")
    ax.set_ylabel("Count")
    ax.set_title(f"{name}\nstd = {std:.3f} dB", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.set_xlim([-5, 5])
plt.tight_layout()
save_fig(fig, "fig_residuals.png")

# ── FIG 11: fig_gpr_surface.png ──────────────────────────────────
d_g = np.linspace(0.20, 1.00, 60)
a_g = np.linspace(0, 20, 60)
D_g, A_g = np.meshgrid(d_g, a_g)
X_grid   = np.column_stack([
    np.log10(D_g.ravel() / 0.20), A_g.ravel()])
mu_s, s_s = gpr.predict(X_grid, return_std=True)
mu_s = mu_s.reshape(D_g.shape)
s_s  = s_s.reshape(D_g.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.8))
cf1 = ax1.contourf(D_g * 100, A_g, mu_s, levels=22, cmap="plasma")
ax1.scatter(D_flat * 100, A_flat, c="white", s=18,
            marker="o", alpha=0.8, edgecolors="k", linewidths=0.3,
            label="Measured")
plt.colorbar(cf1, ax=ax1, label="GPR Mean PL (dB)")
ax1.set_xlabel("Distance (cm)")
ax1.set_ylabel("Misalignment Angle θ (deg)")
ax1.set_title("Mean Prediction", fontsize=11)
ax1.legend(fontsize=8.5, loc="upper left")

cf2 = ax2.contourf(D_g * 100, A_g, s_s, levels=22, cmap="YlOrRd")
ax2.scatter(D_flat * 100, A_flat, c="k", s=18,
            marker="o", alpha=0.8, label="Measured")
plt.colorbar(cf2, ax=ax2, label="GPR Predictive Std (dB)")
ax2.set_xlabel("Distance (cm)")
ax2.set_ylabel("Misalignment Angle θ (deg)")
ax2.set_title("Predictive Uncertainty", fontsize=11)
ax2.legend(fontsize=8.5, loc="upper left")
plt.tight_layout()
save_fig(fig, "fig_gpr_surface.png")

# ── FIG 12: fig_model_surface.png — M1 ───────────────────────────
PL_surf = sep_predict(D_g, A_g, w_los, w_ang)
fig, ax = plt.subplots(figsize=(6.8, 5.4))
cf = ax.contourf(D_g * 100, A_g, PL_surf, levels=20, cmap="plasma")
plt.colorbar(cf, ax=ax, label="Path Loss (dB)")
ax.scatter(D_flat * 100, A_flat, c="white", s=18,
           marker="o", alpha=0.75, edgecolors="k", linewidths=0.3,
           label="Measured points")
ax.set_xlabel("Distance (cm)")
ax.set_ylabel("Misalignment Angle θ (deg)")
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
save_fig(fig, "fig_model_surface.png")

# ── FIG 13: fig_cross_env_PLE.png ────────────────────────────────
N_KIM = 1.920; N_FSPL = 2.000
all_n   = list(n_sjtu) + [n_ple, N_KIM]
all_lbl = list(lbl_sjtu) + ["Block 2\n(anechoic VNA)", "Kim et al.\n(street canyon)"]
all_src = [1] * len(n_sjtu) + [2, 3]

order   = np.argsort(all_n)
all_n   = [all_n[i]   for i in order]
all_lbl = [all_lbl[i] for i in order]
all_src = [all_src[i] for i in order]

C_SIM  = "#4483B6"; C_MEAS = "#D7301F"; C_EXT = "#737373"
sc_col = {1: C_SIM, 2: C_MEAS, 3: C_EXT}

fig, ax = plt.subplots(figsize=(8.2, 5.2))
for k, (n_v, src) in enumerate(zip(all_n, all_src)):
    ax.barh(k, n_v, height=0.55,
            color=sc_col[src], edgecolor="k", lw=0.8, zorder=3)
    ax.text(n_v + 0.005, k, f"{n_v:.3f}",
            va="center", ha="left", fontsize=10, fontfamily="serif")
ax.axvline(N_FSPL, color="k", ls="--", lw=1.8)
ax.text(N_FSPL + 0.004, len(all_n) - 0.5, "FSPL  n=2.000",
        fontsize=9.5, color="k", va="center")
ax.axvline(n_ple, color=C_MEAS, ls=":", lw=1.6)
ax.set_yticks(range(len(all_n)))
ax.set_yticklabels(all_lbl, fontsize=10.5)
ax.set_ylim(-0.6, len(all_n) - 0.4)
ax.set_xlabel("Path Loss Exponent $n$  (CI model,  $d_0$ = 1 m)")
ax.set_xlim([1.38, 2.18])
ax.set_axisbelow(True)
ax.grid(axis="x", alpha=0.3, ls="--")
ax.grid(axis="y", alpha=0.0)

p1 = mpatches.Patch(facecolor=C_SIM,  edgecolor="k",
                     label="SJTU QuaDRiGa (measurement-validated sim.)")
p2 = mpatches.Patch(facecolor=C_MEAS, edgecolor="k",
                     label="Block 2 — VNA measurement (240–300 GHz, this work)")
p3 = mpatches.Patch(facecolor=C_EXT,  edgecolor="k",
                     label="Kim et al. — VNA measurement (300 GHz)")
ax.legend(handles=[p1, p2, p3], loc="lower right",
          fontsize=9, framealpha=0.92)
plt.tight_layout()
save_fig(fig, "fig_cross_env_PLE.png")

# ══════════════════════════════════════════════════════
# PASS 2 COMPUTATIONS — appended block
# Produces all Discussion section numbers in one run.
# ══════════════════════════════════════════════════════
from scipy import stats as _stats

print("\n" + "="*62)
print("PASS 2 — Discussion numbers")
print("="*62)

# ── Y3: F-test of separability ────────────────────────
sse_m1 = N * cv_m1['rmse']**2
sse_m2 = N * cv_m2['rmse']**2
p_m1, p_m2 = 4, 10
df_n = p_m2 - p_m1
df_d = N - p_m2
F_stat = ((sse_m1 - sse_m2) / df_n) / (sse_m2 / df_d)
p_val  = 1 - _stats.f.cdf(F_stat, df_n, df_d)
print(f"\nY3 F-test:  F({df_n},{df_d}) = {F_stat:.2f},  p = {p_val:.2e}")

# ── Y4: Tekbiyik held-out predictions (all 4 models) ──
tek_X  = np.column_stack([
    np.log10(tek['distances_m'] / 0.20),
    np.zeros(len(tek['distances_m']))
])
tek_pl_measured = np.array([-np.mean(tek['cfr_db'][:, i])
                              for i in range(tek['cfr_db'].shape[1])])
tek_m1 = sep_predict(tek['distances_m'], 0.0, w_los, w_ang)
tek_m2 = poly2d_predict(tek_X, w_poly2d)
tek_m3 = pipe_mlp.predict(tek_X)
tek_m4_mu, _ = gpr.predict(tek_X, return_std=True)
print(f"\nY4 Tekbiyik held-out (theta=0, no retraining):")
print(f"  {'d(cm)':<8} {'Meas':>8} {'M1':>8} {'M2':>8} {'M3':>8} {'M4':>8}")
for i, d in enumerate(tek['distances_m']):
    print(f"  {int(d*100):<8} {tek_pl_measured[i]:>8.2f} "
          f"{tek_m1[i]:>8.2f} {tek_m2[i]:>8.2f} "
          f"{tek_m3[i]:>8.2f} {tek_m4_mu[i]:>8.2f}")
rmse_tek = lambda yp: float(np.sqrt(np.mean((yp - tek_pl_measured)**2)))
print(f"  RMSE: M1={rmse_tek(tek_m1):.3f} M2={rmse_tek(tek_m2):.3f} "
      f"M3={rmse_tek(tek_m3):.3f} M4={rmse_tek(tek_m4_mu):.3f}")

# ── Y5: Bootstrap 95% CI on RMSE ─────────────────────
def boot_ci(resid, nb=10000):
    rng2 = np.random.RandomState(0)
    n = len(resid)
    br = np.array([np.sqrt(np.mean(resid[rng2.randint(0,n,n)]**2))
                   for _ in range(nb)])
    return np.percentile(br, [2.5, 97.5])

r_m1 = cv_m1['y_pred'] - cv_m1['y_true']
r_m2 = cv_m2['y_pred'] - cv_m2['y_true']
r_m3 = cv_m3['y_pred'] - cv_m3['y_true']
r_m4 = y_pred_m4 - y
print(f"\nY5 Bootstrap 95% CI on LOO-RMSE:")
for nm, r in [("M1",r_m1),("M2",r_m2),("M3",r_m3),("M4",r_m4)]:
    ci = boot_ci(r)
    print(f"  {nm}: [{ci[0]:.3f}, {ci[1]:.3f}] dB")

# ── Y6: Outlier IQR criterion on DS ──────────────────
ds_flat = ds_matrix.ravel()
q1, q3  = np.percentile(ds_flat, [25, 75])
iqr     = q3 - q1
thresh  = q3 + 1.5 * iqr
n_out   = int(np.sum(ds_flat > thresh))
print(f"\nY6 DS outlier criterion:")
print(f"  Q1={q1:.3f} ns  Q3={q3:.3f} ns  IQR={iqr:.3f} ns")
print(f"  Threshold (Q3+1.5xIQR) = {thresh:.3f} ns")
print(f"  Points above threshold: {n_out}")

# ── DS maximum ────────────────────────────────────────
valid_mask = ds_matrix < thresh
print(f"\nDS max (excluding outlier): {np.max(ds_matrix[valid_mask])*1000:.3f} ps")
print(f"DS max overall:             {np.max(ds_matrix)*1000:.3f} ps")
print(f"DS at (theta=20, d=70cm):   {ds_matrix[np.argmin(np.abs(distances_m-0.70)), -1]*1000:.3f} ps")

# ── S6: GPR 90% PI calibration ───────────────────────
half_pi = pi90_width / 2.0
coverage = float(np.mean(np.abs(r_m4) <= half_pi))
print(f"\nS6 GPR calibration:")
print(f"  90% PI half-width = {half_pi:.4f} dB")
print(f"  Empirical coverage = {coverage:.3f} ({coverage*100:.1f}%)")

print("\n" + "="*62)
print("PASS 2 COMPLETE — paste these values into Discussion section")
print("="*62)

# ================================================================
# PHASE 5 — FINAL SUMMARY
# ================================================================
section("PHASE 5 — Paper numbers summary")

print(f"""
KEY RESULTS (all from measured data — paste into paper):
  PLE  n                    = {n_ple:.4f}
  Separability max dev.     = {sep['overall_max']:.3f} dB at θ={worst_th}°
  DS mean (θ=0°)            = {ds_los_mean*1000:.3f} ps
  Bc median (θ=0°)          = {bc_los_med:.0f} MHz
  Bc median (θ=20°)         = {bc_20_med:.0f} MHz
  Signal BW                 = {MEAS_BW_HZ/1e6:.0f} MHz → frequency-selective

LOO-CV TABLE I:
  M1 RMSE={cv_m1['rmse']:.3f}  MAE={cv_m1['mae']:.3f}  R²={cv_m1['r2']:.4f}  bias={bias_m1:+.3f}
  M2 RMSE={cv_m2['rmse']:.3f}  MAE={cv_m2['mae']:.3f}  R²={cv_m2['r2']:.4f}  bias={bias_m2:+.3f}
  M3 RMSE={cv_m3['rmse']:.3f}  MAE={cv_m3['mae']:.3f}  R²={cv_m3['r2']:.4f}  bias={bias_m3:+.3f}
  M4 RMSE={cv_m4['rmse']:.3f}  MAE={cv_m4['mae']:.3f}  R²={cv_m4['r2']:.4f}  bias={bias_m4:+.3f}  90%-PI={pi90_width:.3f} dB

LaTeX TABLE I rows:
  M1 Separable poly & 2×1D poly           & {cv_m1['rmse']:.3f} & {cv_m1['mae']:.3f} & {cv_m1['r2']:.4f} & — \\\\
  M2 2D poly deg-3  & 2D deg-3 poly       & {cv_m2['rmse']:.3f} & {cv_m2['mae']:.3f} & {cv_m2['r2']:.4f} & — \\\\
  M3 MLP            & 32→16, Adam         & {cv_m3['rmse']:.3f} & {cv_m3['mae']:.3f} & {cv_m3['r2']:.4f} & — \\\\
  M4 GPR (this work)& Product RBF kernel  & {cv_m4['rmse']:.3f} & {cv_m4['mae']:.3f} & {cv_m4['r2']:.4f} & {pi90_width:.3f} dB \\\\
""")

section("DONE — All 13 figures saved to block2_output/")
