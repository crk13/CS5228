"""Guard the analysis joins, comparable experiments and weight-calibration boundary."""

import unittest

import numpy as np
import pandas as pd

from notebooks.Yuxin.compare_temporal import candidates
from notebooks.Yuxin.evaluate_complementarity import align_prediction, calibration_weight, group_rows, score_rows


class AnalysisTests(unittest.TestCase):
    def canonical(self):
        return pd.DataFrame(dict(fold=["holdout"]*2+["rolling_2024_04"]*2,
                                 row_index=[4, 8, 8, 9],
                                 RENT_APPROVAL_DATE=["2023-12", "2024-04", "2024-04", "2024-05"],
                                 MONTHLY_RENT=[1000, 2000, 2000, 3000], REGION=["east"]*4,
                                 TOWN=["a", "b", "b", "c"], FLAT_TYPE=["3-room"]*4,
                                 horizon=[1, 5, 1, 2], horizon_bin=["1-4", "5-8", "1-4", "1-4"]))

    def test_alignment_is_keyed_not_positional_and_rejects_bad_sources(self):
        canonical = self.canonical()
        source = canonical[["fold", "row_index", "RENT_APPROVAL_DATE", "MONTHLY_RENT"]].copy()
        source["Predicted"] = [1100, 1800, 2100, 2800]
        np.testing.assert_array_equal(align_prediction(source.iloc[::-1], canonical), source.Predicted)
        for invalid in (pd.concat([source, source.iloc[:1]]), source.iloc[:-1],
                        source.assign(Predicted=np.nan), source.assign(Predicted=-1)):
            with self.assertRaises(ValueError):
                align_prediction(invalid, canonical)
        with self.assertRaises(AssertionError):
            align_prediction(source.assign(MONTHLY_RENT=999), canonical)
        with self.assertRaises(AssertionError):
            align_prediction(source.assign(RENT_APPROVAL_DATE="2099-01"), canonical)
        altered = source.copy()
        altered.loc[0, "row_index"] = 999
        with self.assertRaises(ValueError):
            align_prediction(altered, canonical)

    def test_group_populations_and_rmse_preserve_overlapping_fold_identity(self):
        canonical = self.canonical()
        prediction = np.array([1100, 1800, 2100, 2800])
        scores = score_rows(canonical, "model", prediction)
        self.assertEqual([r["rows"] for r in scores], [2, 2])
        self.assertAlmostEqual(scores[0]["rmse"], np.sqrt(25000))
        grouped = pd.DataFrame(group_rows(canonical, "model", prediction))
        self.assertTrue(grouped.groupby(["fold", "dimension"]).rows.sum().eq(2).all())
        self.assertEqual(len(grouped.dimension.unique()), 6)

    def test_weight_uses_calibration_only_has_bounds_and_handles_identical_models(self):
        y, a, b = np.array([1., 2., 3.]), np.array([1., 3., 4.]), np.array([0., 1., 2.])
        weight = calibration_weight(y, a, b)
        grid = np.linspace(0, 1, 10001)
        objective = [np.mean((w*a+(1-w)*b-y)**2) for w in grid]
        self.assertAlmostEqual(weight, grid[np.argmin(objective)], places=4)
        self.assertEqual(calibration_weight(y, a, a), .5)
        self.assertEqual(calibration_weight(np.repeat(1e9, 3), a, b), 1.)
        self.assertEqual(calibration_weight(np.repeat(-1e9, 3), a, b), 0.)

    def test_matched_encoders_share_windows_inputs_and_training_budget(self):
        recipes = candidates()
        matched = [(name, p) for name, p, scope in recipes if scope == "matched_L6"]
        self.assertEqual(len(matched), 5)
        reference = {k: v for k, v in matched[0][1].items() if k != "kind"}
        for _, parameters in matched:
            self.assertEqual({k: v for k, v in parameters.items() if k != "kind"}, reference)
        self.assertEqual(reference["lookback"], 6)
        self.assertEqual(sum(p["kind"] == "transformer" for _, p, _ in recipes), 8)


if __name__ == "__main__":
    unittest.main()
