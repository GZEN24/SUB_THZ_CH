"""Reproduce the headline validation numbers (LOO, LODO, LOAO, 50/80 cm holdout, M1-M5).

Usage (from repository root):
    python code/reproduce_validation.py                 # everything
    python code/reproduce_validation.py --skip-loo      # grouped protocols only (faster)
    python code/reproduce_validation.py --skip-mlp      # omit M3 (neural network)

Outputs go to results/reproduced/. The final block compares against the values
reported in the paper (data/expected_headline_numbers.csv) and prints PASS/FAIL
per cell using a tolerance of 0.002 dB (RMSE) / 0.5 percentage points (coverage).
Tiny differences can arise from library versions: see environment/README.md.
"""
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import subthz_pl as S

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reproduced"
GP_RESTARTS = 2          # per fit (paper: two restarts plus the initial optimization)


def predict_fold(model, df, X, y, train, test, protocol, label):
    if model == "M1":
        return S.m1_predict_group(df, train, test, theta0_held_out=(protocol == "LOAO" and label == "0deg")), None
    if model == "M2":
        return S.m2_fit_predict(X[train], y[train], X[test]), None
    if model == "M3":
        return S.m3_fit_predict(X[train], y[train], X[test]), None
    return S.gp_fit_predict(model, X[train], y[train], X[test], GP_RESTARTS)


def rows_for(df, y, idx, mu, sd, model, protocol, fold):
    r = pd.DataFrame({"row_id": df.row_id.iloc[idx].to_numpy(), "distance_m": df.distance_m.iloc[idx].to_numpy(),
                      "angle_deg": df.angle_deg.iloc[idx].to_numpy(), "measured_pl_db": y[idx],
                      "predicted_pl_db": mu, "model": model, "protocol": protocol, "fold": fold})
    r["residual_db"] = r.predicted_pl_db - r.measured_pl_db
    if sd is not None:
        h = S.Z90 * sd
        r["predictive_std_db"] = sd
        r["covered"] = np.abs(r.residual_db) <= h
        r["full_width_db"] = 2 * h
        r["interval_score_db"] = S.interval_score(y[idx], mu, sd)
    return r


def run_grouped(df, X, y, models):
    out = []
    for protocol, label, train, test in S.folds(df):
        for m in models:
            mu, sd = predict_fold(m, df, X, y, train, test, protocol, label)
            out.append(rows_for(df, y, test, np.asarray(mu), sd, m, protocol, label))
        print(f"  {protocol} {label} done", flush=True)
    return pd.concat(out, ignore_index=True)


def run_loo(df, X, y, models):
    out = []
    n = len(y)
    for m in models:
        if m == "M1":
            mu = S.m1_loo(df)
            out.append(rows_for(df, y, np.arange(n), mu, None, m, "LOO", "pointwise"))
        else:
            mus, sds = np.zeros(n), np.zeros(n)
            for i in range(n):
                tr = np.setdiff1d(np.arange(n), [i])
                if m == "M2":
                    mus[i] = S.m2_fit_predict(X[tr], y[tr], X[[i]])[0]
                elif m == "M3":
                    mus[i] = S.m3_fit_predict(X[tr], y[tr], X[[i]])[0]
                else:
                    a, b = S.gp_fit_predict(m, X[tr], y[tr], X[[i]], GP_RESTARTS)
                    mus[i], sds[i] = a[0], b[0]
            out.append(rows_for(df, y, np.arange(n), mus, sds if m in ("M4", "M5") else None, m, "LOO", "pointwise"))
        print(f"  LOO {m} done", flush=True)
    return pd.concat(out, ignore_index=True)


def summarize(pred):
    rec = []
    for (m, p), g in pred.groupby(["model", "protocol"], sort=False):
        fold_rmse = g.groupby("fold", sort=False).residual_db.apply(S.rmse)
        rec.append(dict(model=m, protocol=p, pooled_rmse_db=S.rmse(g.residual_db),
                        mean_fold_rmse_db=float(fold_rmse.mean()), n_points=len(g),
                        coverage_pct=100 * float(g.covered.mean()) if "covered" in g and g.covered.notna().all() else np.nan))
    return pd.DataFrame(rec)


def compare(summary, expected=None):
    if expected is None:
        expected = pd.read_csv(ROOT / "data" / "expected_headline_numbers.csv")
    keys = ["model", "protocol", "metric"]
    if expected.duplicated(keys).any():
        raise ValueError("Duplicate expected headline metric")
    if summary.duplicated(["model", "protocol"]).any():
        raise ValueError("Duplicate reproduced model/protocol result")
    res = []
    for e in expected.itertuples(index=False):
        s = summary[(summary.model == e.model) & (summary.protocol == e.protocol)]
        got = float(s[e.metric].iloc[0]) if not s.empty and e.metric in s else np.nan
        tol = 0.5 if e.metric == "coverage_pct" else 0.002
        diff = abs(got - e.value)
        res.append(dict(model=e.model, protocol=e.protocol, metric=e.metric, paper=e.value,
                        reproduced=round(got, 4), abs_diff=round(diff, 4),
                        status="PASS" if np.isfinite(diff) and diff <= tol else "FAIL"))
    return pd.DataFrame(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-loo", action="store_true")
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--skip-grouped", action="store_true")
    a = ap.parse_args()
    if a.skip_grouped and a.skip_loo:
        ap.error("--skip-grouped and --skip-loo cannot be used together")
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    df, X, y = S.load_targets()
    models = ["M1", "M2"] + ([] if a.skip_mlp else ["M3"]) + ["M4", "M5"]
    parts = []
    if not a.skip_grouped:
        print("Grouped protocols (LODO / LOAO / 50-80 cm)")
        parts.append(run_grouped(df, X, y, models))
    if not a.skip_loo:
        print("Pointwise LOO")
        parts.append(run_loo(df, X, y, models))
    pred = pd.concat(parts, ignore_index=True)
    pred.to_csv(OUT / "predictions.csv", index=False, float_format="%.12g")
    summ = summarize(pred)
    summ.to_csv(OUT / "summary.csv", index=False, float_format="%.6g")
    expected = pd.read_csv(ROOT / "data" / "expected_headline_numbers.csv")
    selected_protocols = ({"LOO"} if a.skip_grouped else
                          {"LODO", "LOAO", "HOLDOUT_50_80"} if a.skip_loo else
                          {"LOO", "LODO", "LOAO", "HOLDOUT_50_80"})
    expected = expected[expected.model.isin(models) & expected.protocol.isin(selected_protocols)]
    cmp_ = compare(summ, expected)
    cmp_.to_csv(OUT / "comparison_with_paper.csv", index=False)
    print("\n", cmp_.to_string(index=False))
    passed = int((cmp_.status == "PASS").sum())
    failed = int((cmp_.status == "FAIL").sum())
    print(f"\n{passed} PASS / {failed} FAIL ({len(expected)} selected paper checks)")
    if failed or not len(expected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
