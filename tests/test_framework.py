"""Small contract tests; no competition data is needed."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from threadpoolctl import threadpool_limits

from src.config import ROOT_DIR, SEED
from src.cv import DATE_COL, get_splits
from src.data_cleaning import NUM_COLS, CAT_COLS, TARGET, ID_COL
from src.experiment import run_experiment, write_submission
from src.features import FeatureBlock, FeaturePipeline, FEATURE_REGISTRY, BLOCK_COLUMNS
from src.models import ModelWrapper, RecentMedianRegressor, TargetTransform, hist_gradient_boosting_model


def example_train():
    dates = ["2021-01", "2023-06", "2023-11", "2023-12", "2024-03", "2024-04", "2025-03"]
    df = pd.DataFrame({c: [1.0] * len(dates) for c in NUM_COLS})
    for column in CAT_COLS:
        df[column] = pd.Series(["known"] * len(dates), dtype="string")
    df[DATE_COL] = pd.Series(dates, dtype="string")
    df["RENT_YEAR"] = [int(d[:4]) for d in dates]
    df["RENT_MONTH"] = [int(d[5:]) for d in dates]
    df["FLOOR_AREA_SQM"] = 80.0
    df["YEAR_COMPLETED"] = 1990.0
    df["LEASE_COMMENCE_DATE"] = 1992
    df["FLAT_TYPE"] = "4-room"
    df[TARGET] = [1800, 2400, 2700, 3000, 3200, 3300, 3400]
    df.index = np.arange(len(df)) * 10 + 3
    return df


class LearnedMeans(FeatureBlock):
    fits = []
    transforms = []

    def fit(self, train_df):
        type(self).fits.append(train_df.copy())
        self.means_ = train_df.groupby("TOWN")[TARGET].mean()
        return self

    def transform(self, df):
        if TARGET in df or ID_COL in df:
            raise AssertionError("transform received target or ID")
        type(self).transforms.append(df.index.copy())
        return pd.DataFrame({"LEARNED_MEAN": df["TOWN"].map(self.means_)}, index=df.index)


class EchoRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X.iloc[:, 0].to_numpy(dtype=float)


class FrameworkTests(unittest.TestCase):
    def test_shared_splits_are_chronological_and_positional(self):
        dates = pd.period_range("2021-01", "2026-07", freq="M").astype(str)
        df = pd.DataFrame({DATE_COL: dates}).iloc[::-1]
        holdout = get_splits(df, "holdout")
        rolling = get_splits(df, "rolling")
        self.assertEqual(len(holdout), 1)
        self.assertEqual(len(rolling), 2)
        np.testing.assert_array_equal(holdout[0].train_idx, rolling[0].train_idx)
        for fold, train_end, valid_start, valid_rows in zip(
            rolling, ["2023-11", "2024-03"], ["2023-12", "2024-04"], [16, 12]
        ):
            training = df.iloc[fold.train_idx][DATE_COL]
            validation = df.iloc[fold.valid_idx][DATE_COL]
            self.assertEqual(training.max(), train_end)
            self.assertEqual(validation.min(), valid_start)
            self.assertEqual(validation.max(), "2025-03")
            self.assertEqual(len(validation), valid_rows)
            self.assertLess(training.max(), validation.min())
            self.assertFalse(set(fold.train_idx).intersection(fold.valid_idx))
        with self.assertRaises(ValueError):
            get_splits(df, "random_kfold")
        with self.assertRaises(ValueError):
            get_splits(df.iloc[:1])

    def test_feature_switches_and_reserved_blocks(self):
        train = example_train()
        basic = FeaturePipeline(["basic"]).fit_transform(train)
        combined = FeaturePipeline(["basic", "block"]).fit_transform(train)
        self.assertTrue(basic.index.equals(train.index))
        self.assertNotIn(TARGET, combined)
        self.assertTrue(set(BLOCK_COLUMNS).isdisjoint(basic.columns))
        self.assertTrue(set(BLOCK_COLUMNS) <= set(combined.columns))
        self.assertEqual(basic["MONTH_INDEX"].iloc[0], 0)
        self.assertEqual(basic["MONTH_INDEX"].iloc[-1], 50)
        with self.assertRaises(NotImplementedError):
            FeaturePipeline(["target_enc"])

    def test_validation_targets_never_reach_feature_transform(self):
        train = example_train()
        changed = train.copy()
        fold = get_splits(train)[0]
        changed.iloc[fold.valid_idx, changed.columns.get_loc(TARGET)] = 1_000_000
        LearnedMeans.fits = []
        LearnedMeans.transforms = []
        model = ModelWrapper(EchoRegressor(), ["LEARNED_MEAN"])
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as directory, patch.dict(
            FEATURE_REGISTRY, {"fake_target": LearnedMeans}
        ):
            with patch("src.experiment.load_train", return_value=train):
                first = run_experiment("leakage", "test", ["fake_target"], model, output_dir=directory)
            with patch("src.experiment.load_train", return_value=changed):
                second = run_experiment("leakage", "test", ["fake_target"], model, output_dir=directory)
            predictions = [pd.read_csv(r.run_dir / "validation_predictions.csv") for r in (first, second)]
            np.testing.assert_array_equal(predictions[0]["Predicted"], predictions[1]["Predicted"])
            np.testing.assert_array_equal(predictions[0]["row_index"], fold.valid_idx)
            self.assertNotEqual(first.mean_rmse, second.mean_rmse)
            log = pd.read_csv(Path(directory) / "experiments" / "log.csv")
            self.assertEqual(len(log), 2)
            self.assertEqual(log["run_id"].nunique(), 2)
            config = json.loads((first.run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["seed"], SEED)
        self.assertEqual(len(LearnedMeans.fits), 2)
        for fitted in LearnedMeans.fits:
            pd.testing.assert_frame_equal(fitted, train.iloc[fold.train_idx])
        self.assertFalse(hasattr(model, "estimator_"))

    def test_final_refits_features_on_all_train_and_writes_submission(self):
        train = example_train()
        test = pd.concat([train.iloc[[0]].drop(columns=TARGET)] * 50000, ignore_index=True)
        test[ID_COL] = np.arange(50000)
        test[DATE_COL] = "2026-07"
        test["RENT_YEAR"] = 2026
        test["RENT_MONTH"] = 7
        LearnedMeans.fits = []
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as directory, patch.dict(
            FEATURE_REGISTRY, {"fake_target": LearnedMeans}
        ), patch("src.experiment.load_train", return_value=train), patch(
            "src.experiment.load_test", return_value=test
        ):
            result = run_experiment("final", "test", ["fake_target"],
                                    ModelWrapper(EchoRegressor(), ["LEARNED_MEAN"]),
                                    final=True, output_dir=directory)
            submission = pd.read_csv(result.submission_path)
            self.assertEqual(list(submission.columns), ["Id", "Predicted"])
            self.assertEqual(len(submission), 50000)
            np.testing.assert_array_equal(submission["Id"], test[ID_COL])
            np.testing.assert_allclose(submission["Predicted"], train[TARGET].mean())
            self.assertTrue(np.isfinite(submission["Predicted"]).all())
        self.assertEqual(len(LearnedMeans.fits), 2)
        pd.testing.assert_frame_equal(LearnedMeans.fits[-1], train)

    def test_malformed_submission_is_rejected(self):
        test = pd.DataFrame({ID_COL: np.arange(50000)})
        with tempfile.TemporaryDirectory(dir=ROOT_DIR) as directory:
            path = Path(directory) / "submission.csv"
            with self.assertRaises(ValueError):
                write_submission(test.iloc[:-1], np.ones(49999), path)
            with self.assertRaises(ValueError):
                write_submission(test.iloc[::-1], np.ones(50000), path)
            with self.assertRaises(ValueError):
                write_submission(test, np.full(50000, np.nan), path)
            self.assertFalse(path.exists())

    def test_target_transform_predictions_are_original_scale(self):
        X = pd.DataFrame({"FLOOR_AREA_SQM": [50.0, 100.0]})
        for transform, targets, expected in (
            ("none", [1000, 4000], [2500, 2500]),
            ("log", [1000, 4000], [2000, 2000]),
            ("per_sqm", [2500, 5000], [2500, 5000]),
        ):
            model = ModelWrapper(DummyRegressor(), ["FLOOR_AREA_SQM"], transform).fit(X, targets)
            np.testing.assert_allclose(model.predict(X), expected)
        with self.assertRaises(ValueError):
            TargetTransform("per_sqm").transform([1000], pd.DataFrame({"FLOOR_AREA_SQM": [0]}))

    def test_recent_median_uses_six_calendar_months_and_fallbacks(self):
        X = pd.DataFrame({
            DATE_COL: ["2023-06", "2023-07", "2023-12", "2023-12"],
            "TOWN": ["a", "a", "a", "b"], "FLAT_TYPE": ["4-room"] * 4,
        })
        fitted = RecentMedianRegressor().fit(X, [100, 3000, 4000, 5000])
        query = pd.DataFrame({"TOWN": ["a", "unseen", "unseen"],
                              "FLAT_TYPE": ["4-room", "4-room", "unseen"]})
        np.testing.assert_array_equal(fitted.predict(query), [3500, 4000, 4000])

    def test_hist_model_handles_unseen_categories_and_missing_building_values(self):
        train = example_train()
        pipeline = FeaturePipeline(["basic", "block"])
        X = pipeline.fit_transform(train)
        validation = train.iloc[[0]].copy()
        validation["TOWN"] = "unseen town"
        validation["YEAR_COMPLETED"] = np.nan
        validation["MAX_FLOOR"] = np.nan
        model = hist_gradient_boosting_model()
        model.set_params(estimator__regressor__max_iter=2, estimator__regressor__min_samples_leaf=2)
        with threadpool_limits(limits=1):
            model.fit(X, train[TARGET])
            prediction = model.predict(pipeline.transform(validation))
        self.assertEqual(prediction.shape, (1,))
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
