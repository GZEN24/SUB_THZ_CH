"""Joint distance-angle path-loss models M1-M5 for the 240-300 GHz VNA dataset.

Self-contained re-implementation of the models used in the paper. It reads only
the processed 153-point table in ``data/pl_targets_153.csv`` (no raw MAT files).

Features (deterministic, no fitted scaler for the polynomial/GP models):
    x1 = log10(d / 0.20 m),  x2 = theta / 20 deg
Target: measured path loss in dB (negative arithmetic mean of 4096 CFR dB values).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
Z90 = 1.645            # z-value used for nominal 90 % intervals (paper convention)
ANGLE_CAP_DEG = 20.0
POLY_DEGREE = 3
DATA = Path(__file__).resolve().parents[1] / "data" / "pl_targets_153.csv"


# ----------------------------------------------------------------- data ----
def load_targets(path=DATA):
    """Return (dataframe, X, y). Rows are distance-major: 9 distances x 17 angles."""
    df = pd.read_csv(path, float_precision="round_trip")
    required = {"row_id", "distance_m", "angle_deg", "measured_pl_db"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing target columns: {sorted(required - set(df.columns))}")
    expected_distances = np.round(np.arange(0.2, 1.01, 0.1), 10)
    expected_angles = np.r_[np.arange(16), 20]
    if (len(df) != 153
            or not np.array_equal(df.row_id.to_numpy(), np.arange(153))
            or not np.allclose(df.distance_m.to_numpy(), np.repeat(expected_distances, 17), rtol=0, atol=1e-9)
            or not np.array_equal(df.angle_deg.to_numpy(), np.tile(expected_angles, 9))
            or not np.isfinite(df.measured_pl_db.to_numpy()).all()):
        raise ValueError("Expected the complete finite 9 x 17 grid in canonical distance-major row order")
    X = np.column_stack([np.log10(df.distance_m.to_numpy() / 0.2),
                         df.angle_deg.to_numpy() / ANGLE_CAP_DEG])
    return df, X, df.measured_pl_db.to_numpy()


def folds(df):
    """Grouped folds on distance-major indices: LODO (9), LOAO (17), 50/80 cm holdout."""
    for protocol, col in (("LODO", "distance_m"), ("LOAO", "angle_deg")):
        for value in sorted(df[col].unique()):
            test = np.flatnonzero(np.isclose(df[col], value))
            label = f"{int(round(value * 100))}cm" if col == "distance_m" else f"{int(value)}deg"
            yield protocol, label, np.setdiff1d(np.arange(len(df)), test), test
    test = np.flatnonzero(np.isclose(df.distance_m, .5) | np.isclose(df.distance_m, .8))
    yield "HOLDOUT_50_80", "50_80cm", np.setdiff1d(np.arange(len(df)), test), test


# ------------------------------------------------------------------ M1 ----
def fit_separable(d, a, pl):
    """M1: PL = alpha*log10(d)+beta (theta=0 column) + (a*theta^2 + b*theta).

    d, a : sorted unique training distances [m] / angles [deg]; pl : (len(d), len(a)).
    Requires a[0] == 0 (use fit_separable_no_los otherwise).
    """
    ld = np.log10(d)
    w_los = np.linalg.lstsq(np.column_stack([ld, np.ones_like(ld)]), pl[:, 0], rcond=None)[0]
    delta = np.mean(pl - (w_los[0] * ld + w_los[1])[:, None], axis=0)
    th = np.asarray(a, float)
    w_ang = np.linalg.lstsq(np.column_stack([th ** 2, th]), delta, rcond=None)[0]
    return w_los, w_ang


def fit_separable_no_los(d, a, pl):
    """M1 variant used only when the theta=0 column is withheld (LOAO fold 0 deg)."""
    ld = np.log10(d)
    alpha = float(np.mean([np.linalg.lstsq(np.column_stack([ld, np.ones_like(ld)]),
                                           pl[:, j], rcond=None)[0][0] for j in range(len(a))]))
    c = np.array([np.mean(pl[:, j] - alpha * ld) for j in range(len(a))])
    th = np.asarray(a, float)
    coef = np.linalg.lstsq(np.column_stack([th ** 2, th, np.ones_like(th)]), c, rcond=None)[0]
    return np.array([alpha, coef[2]]), np.array([coef[0], coef[1]])


def sep_predict(d, theta, w_los, w_ang):
    d, theta = np.asarray(d, float), np.asarray(theta, float)
    return w_los[0] * np.log10(d) + w_los[1] + w_ang[0] * theta ** 2 + w_ang[1] * theta


def m1_predict_group(df, train, test, theta0_held_out=False):
    """Fit M1 on a rectangular training sub-grid and predict the test rows."""
    sub = df.iloc[train]
    d, a = np.sort(sub.distance_m.unique()), np.sort(sub.angle_deg.unique())
    pl = sub.measured_pl_db.to_numpy().reshape(len(d), len(a))
    w_los, w_ang = (fit_separable_no_los if theta0_held_out else fit_separable)(d, a, pl)
    return sep_predict(df.iloc[test].distance_m, df.iloc[test].angle_deg, w_los, w_ang)


def m1_loo(df):
    """Pointwise LOO for M1 (cell removal; LOS fit drops the row only for theta=0 cells)."""
    d, a = np.sort(df.distance_m.unique()), np.sort(df.angle_deg.unique())
    nd, na = len(d), len(a)
    pl = df.measured_pl_db.to_numpy().reshape(nd, na)
    ld, th = np.log10(d), a.astype(float)
    out = np.zeros(nd * na)
    for idx in range(nd * na):
        i, j = divmod(idx, na)
        m = np.ones(nd, bool)
        if j == 0:
            m[i] = False
        w_l = np.linalg.lstsq(np.column_stack([ld[m], np.ones(m.sum())]), pl[m, 0], rcond=None)[0]
        base = w_l[0] * ld + w_l[1]
        delta = np.array([np.mean([pl[r, c] - base[r] for r in range(nd) if not (r == i and c == j)])
                          for c in range(na)])
        w_a = np.linalg.lstsq(np.column_stack([th ** 2, th]), delta, rcond=None)[0]
        out[idx] = sep_predict(d[i], a[j], w_l, w_a)
    return out


# ------------------------------------------------------------------ M2 ----
def _phi(X, degree=POLY_DEGREE):
    terms = [(i, t - i) for t in range(degree + 1) for i in range(t + 1)]
    return np.column_stack([X[:, 0] ** i * X[:, 1] ** j for i, j in terms])


def m2_fit_predict(Xtr, ytr, Xte):
    return _phi(Xte) @ np.linalg.lstsq(_phi(Xtr), ytr, rcond=None)[0]


# ------------------------------------------------------------------ M3 ----
def build_mlp(hidden=(32, 16), alpha=0.01, seed=SEED):
    mlp = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=alpha,
                       learning_rate="adaptive", max_iter=3000, early_stopping=True,
                       validation_fraction=0.15, n_iter_no_change=50, random_state=seed, tol=1e-5)
    return Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])


def m3_fit_predict(Xtr, ytr, Xte, **kw):
    return build_mlp(**kw).fit(Xtr, ytr).predict(Xte)


# ---------------------------------------------------------------- M4/M5 ----
def gp(model, restarts):
    """M4: RBF; M5: anisotropic radial Matern 5/2. Both: C * k + White, normalize_y=True."""
    if model == "M4":
        k = RBF([.3, 8.], [(.01, 5.), (.5, 100.)])
    elif model == "M5":
        k = Matern([.3, 8.], [(.01, 5.), (.05, 100.)], nu=2.5)
    else:
        raise ValueError(model)
    return GaussianProcessRegressor(kernel=ConstantKernel(10., (1., 1000.)) * k + WhiteKernel(.1, (1e-4, 10.)),
                                    alpha=0., normalize_y=True, n_restarts_optimizer=restarts,
                                    random_state=SEED)


def gp_fit_predict(model, Xtr, ytr, Xte, restarts):
    g = gp(model, restarts).fit(Xtr, ytr)
    mu, sd = g.predict(Xte, return_std=True)   # sd includes the WhiteKernel term
    return mu, sd


# ------------------------------------------------------------- metrics ----
def rmse(err):
    err = np.asarray(err, float)
    return float(np.sqrt(np.mean(err ** 2)))


def interval_score(y, mu, sd, z=Z90):
    h = z * np.asarray(sd)
    lo, hi = mu - h, mu + h
    return 2 * h + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
