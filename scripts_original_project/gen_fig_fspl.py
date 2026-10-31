# ================================================================
# Script       : gen_fig_fspl.py
# Purpose      : Generate the CI model fit + FSPL validation figure.
#                Shows n=1.897 fit against Karakoca θ=0° and
#                Tekbiyik independent validation. Saves as
#                block2_output/fig_fspl_ci_fit.png
#
# Run from same folder as block2_complete.py and block2_master_analysis.py
# ================================================================

from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from block2_complete import (
    load_all_karakoca,
    load_tekbiyik,
    fspl_db,
    fit_separable,
    extract_statistics,
    ALL_ANGLES_DEG,
    ANGLE_CAP_DEG,
    DISTANCES_M,
    TEKBIYIK_DISTS_M,
    F_CENTER_HZ,
    C_LIGHT_MS,
    COLOR,
)

# ── Config ────────────────────────────────────────────────────────
BASE_DIR = Path(
    r"PATH/TO/Datasets"  # EDIT: folder containing Sub_THz_measurements/ (see README)
)
KARAKOCA_DIR  = BASE_DIR / "Sub_THz_measurements"
TEKBIYIK_FILE = (BASE_DIR / "THz_Wireless_Measurements"
                 / "THz_measurement_channel_frequency_response.mat")
OUTPUT_DIR    = SCRIPT_DIR / "block2_output"
OUTPUT_DIR.mkdir(exist_ok=True)

matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "DejaVu Serif"],
    "font.size":       10,
    "axes.labelsize":  11,
    "axes.titlesize":  10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.grid":       True,
    "grid.alpha":      0.2,
})

# ── Load data ─────────────────────────────────────────────────────
print("Loading Karakoca ...")
data_list = load_all_karakoca(KARAKOCA_DIR)
print("Loading Tekbiyik ...")
tek = load_tekbiyik(TEKBIYIK_FILE)

# ── Extract statistics ────────────────────────────────────────────
distances_m, angles_deg, pl_matrix, _, _ = extract_statistics(
    data_list, angle_cap=ANGLE_CAP_DEG)

# ── Fit M1 to get w_los ──────────────────────────────────────────
w_los, w_ang = fit_separable(distances_m, angles_deg, pl_matrix)
n_ple        = w_los[0] / 10.0      # PLE = alpha/10
beta         = w_los[1]

print(f"PLE n = {n_ple:.4f}")
print(f"PL_LOS(d) = {w_los[0]:.4f}·log10(d) + {beta:.4f}")

# ── Tekbiyik path loss ────────────────────────────────────────────
tek_pl = np.array([-np.mean(tek["cfr_db"][:, i])
                    for i in range(tek["cfr_db"].shape[1])])

# ── Reference point: d0 = 20 cm ──────────────────────────────────
d0_m     = 0.20
pl_los   = pl_matrix[:, 0]          # Karakoca θ=0° absolute PL
pl0_kara = pl_los[0]                # PL at 20 cm (absolute)
pl0_tek  = tek_pl[0]                # Tekbiyik at 20 cm

# Relative path loss (dB above reference at 20 cm)
rel_kara = pl_los - pl0_kara
rel_tek  = tek_pl - pl0_tek

# CI model — relative
d_fine   = np.linspace(0.20, 1.00, 300)
log_d    = np.log10(d_fine)
log_d0   = np.log10(d0_m)
rel_ci   = w_los[0] * (log_d - log_d0)   # = 10n·log10(d/d0)

# FSPL relative
lam      = C_LIGHT_MS / F_CENTER_HZ
fspl_abs = 20.0 * np.log10(4.0 * np.pi * d_fine / lam)
fspl_ref = 20.0 * np.log10(4.0 * np.pi * d0_m  / lam)
rel_fspl = fspl_abs - fspl_ref

# ── Figure ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.0, 4.5))

# Karakoca measurements
ax.plot(distances_m * 100, rel_kara,
        "o-", color=COLOR["measured"],
        ms=6, lw=1.5, zorder=5,
        label="Karakoca LoS ($\\theta = 0°$)")

# Tekbiyik validation
ax.plot(TEKBIYIK_DISTS_M * 100, rel_tek,
        "s--", color=COLOR["tekbiyik"],
        ms=6, lw=1.4, zorder=5,
        label="Tekbiyik LoS (validation)")

# CI model fit
ax.plot(d_fine * 100, rel_ci,
        color=COLOR["separable"], ls="--", lw=1.8,
        label=f"CI fit: $n = {n_ple:.3f}$")

# FSPL reference (n=2)
ax.plot(d_fine * 100, rel_fspl,
        color=COLOR["fspl"], ls="-.", lw=1.4,
        label="FSPL: $n = 2.000$ (270 GHz)")

ax.set_xlabel("Distance (cm)")
ax.set_ylabel(f"Relative Path Loss (dB, ref. {int(d0_m*100)} cm)")
ax.set_xlim([15, 105])
ax.set_title("CI Path Loss Model Fit — Measured vs. Analytical\n"
             f"$\\mathrm{{PL}}_{{\\mathrm{{LoS}}}}(d) = "
             f"{w_los[0]:.4f}\\,\\log_{{10}}(d) + {beta:.4f}$ [dB]",
             fontsize=9.5)
ax.legend(loc="upper left", fontsize=8.5)
plt.tight_layout()

out = OUTPUT_DIR / "fig_fspl_ci_fit.png"
fig.savefig(str(out), dpi=300)
plt.close(fig)
print(f"\nSaved: {out}")

# ── Tekbiyik agreement check ──────────────────────────────────────
# Compare relative PL at overlapping distances
overlap_d = [0.20, 0.30, 0.40, 0.60, 0.80]
print("\nKarakoca vs Tekbiyik agreement (relative PL):")
print(f"  {'d(cm)':<8} {'Karakoca':>10} {'Tekbiyik':>10} {'Delta':>8}")
print("  " + "-"*40)
for d_t in overlap_d:
    idx_k = int(np.argmin(np.abs(distances_m - d_t)))
    idx_t = int(np.argmin(np.abs(TEKBIYIK_DISTS_M - d_t)))
    if idx_t >= len(rel_tek):
        continue
    rk = rel_kara[idx_k]
    rt = rel_tek[idx_t]
    print(f"  {int(d_t*100):<8} {rk:>10.3f} {rt:>10.3f} {abs(rk-rt):>8.3f} dB")
