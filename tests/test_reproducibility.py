"""Fast guardrails for the public reproduction package; no model refitting."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import subthz_pl as S
import reproduce_validation as R


class ReproductionTests(unittest.TestCase):
    def test_target_grid_is_complete_and_ordered(self):
        df, X, y = S.load_targets()
        self.assertEqual((len(df), X.shape, y.shape), (153, (153, 2), (153,)))
        self.assertEqual(len(list(S.folds(df))), 27)

    def test_reject_shuffled_grid(self):
        df, _, _ = S.load_targets()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            df.iloc[::-1].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "canonical distance-major"):
                S.load_targets(path)

    def test_comparison_flags_missing_and_nonfinite_metrics(self):
        exp = pd.DataFrame([{"model": "M1", "protocol": "LOO", "metric": "pooled_rmse_db", "value": 0.411}])
        missing = pd.DataFrame([{"model": "M2", "protocol": "LOO", "pooled_rmse_db": 0.411}])
        self.assertEqual(R.compare(missing, exp).status.iloc[0], "FAIL")
        nan = pd.DataFrame([{"model": "M1", "protocol": "LOO", "pooled_rmse_db": np.nan}])
        self.assertEqual(R.compare(nan, exp).status.iloc[0], "FAIL")
        good = pd.DataFrame([{"model": "M1", "protocol": "LOO", "pooled_rmse_db": 0.4111}])
        self.assertEqual(R.compare(good, exp).status.iloc[0], "PASS")
        bad = pd.DataFrame([{"model": "M1", "protocol": "LOO", "pooled_rmse_db": 0.45}])
        self.assertEqual(R.compare(bad, exp).status.iloc[0], "FAIL")

    def test_duplicate_reference_rejected(self):
        exp = pd.DataFrame([{"model": "M1", "protocol": "LOO", "metric": "pooled_rmse_db", "value": .411}])
        with self.assertRaisesRegex(ValueError, "Duplicate expected"):
            R.compare(pd.DataFrame(), pd.concat([exp, exp]))

    def test_m1_zero_angle_group_special_case(self):
        df, _, _ = S.load_targets()
        fold = next(f for f in S.folds(df) if f[:2] == ("LOAO", "0deg"))
        pred = S.m1_predict_group(df, fold[2], fold[3], theta0_held_out=True)
        self.assertEqual(pred.shape, (9,))
        self.assertTrue(np.isfinite(pred).all())


if __name__ == "__main__":
    unittest.main()
