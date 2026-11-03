"""Meaningful invariants for neural inputs, fixed-origin forecasts and shared CV."""

import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone

from src.cv import get_splits
from src.data_cleaning import CAT_COLS, NUM_COLS
from src.experiment import run_experiment
from src.features import FeaturePipeline
from src.models.yuxin_nn import MarketHybridRegressor, TabularNNRegressor
from src.models.yuxin_nn.datasets import monthly_windows
from src.models.yuxin_nn.ensemble import NeuralFamilyEnsemble
from src.models.yuxin_nn.market import MarketForecaster, monthly_market
from src.models.yuxin_nn.preprocessing import TabularPreprocessor
from src.models.yuxin_nn.temporal import MarketEncoder


def fixture():
    dates = np.repeat(pd.period_range("2021-01", "2025-03", freq="M").astype(str), 3)
    n = len(dates)
    frame = pd.DataFrame({c: np.ones(n) for c in NUM_COLS})
    for column in CAT_COLS:
        frame[column] = pd.Series(["known"] * n, dtype="string")
    frame["RENT_APPROVAL_DATE"] = dates
    frame["RENT_YEAR"] = [int(d[:4]) for d in dates]
    frame["RENT_MONTH"] = [int(d[5:]) for d in dates]
    frame["YEAR_COMPLETED"] = 1990.0
    frame["LEASE_COMMENCE_DATE"] = 1992.0
    frame["FLOOR_AREA_SQM"] = np.tile([60., 80., 100.], 51)
    frame["FLAT_TYPE"] = "4-room"
    frame["LATITUDE"], frame["LONGITUDE"] = 1.35, 103.8
    frame["MONTHLY_RENT"] = 1800 + np.repeat(np.arange(51) * 20., 3) + np.tile([-200, 0, 200], 51)
    frame.index = np.arange(n) * 7 + 3
    return frame


def tiny(kind):
    if kind in ("mlp", "transformer"):
        return TabularNNRegressor(kind, epochs=2, batch_size=64, hidden_size=16,
                                  d_model=8, nhead=2, layers=1, inner_months=2, threads=1)
    return MarketHybridRegressor(encoder="transformer" if kind == "temporal_transformer" else kind,
                                 property_epochs=2, property_hidden_size=16,
                                 temporal_epochs=3, batch_size=64, hidden_size=4, threads=1)


class NeuralTests(unittest.TestCase):
    def test_vocab_and_numeric_stats_are_training_only(self):
        train = fixture().iloc[:10]
        X = FeaturePipeline(["basic", "block"]).fit_transform(train)
        X.loc[X.index[:2], "YEAR_COMPLETED"] = np.nan
        processor = TabularPreprocessor(min_frequency=1).fit(X)
        center = processor.mean_.copy()
        query = X.iloc[:2].copy()
        query["TOWN"] = "unseen"
        query["STREET"] = pd.NA
        query["FLOOR_AREA_SQM"] = 100000.
        numeric, categorical = processor.transform(query)
        self.assertTrue(np.isfinite(numeric).all())
        self.assertEqual(numeric.dtype, np.float32)
        np.testing.assert_array_equal(categorical[:, 0], 0)
        np.testing.assert_array_equal(categorical[:, 3], 1)
        np.testing.assert_array_equal(processor.mean_, center)

    def test_monthly_aggregation_is_positional_and_missing_month_is_causal(self):
        X = pd.DataFrame({"RENT_APPROVAL_DATE": ["2021-01", "2021-01", "2021-03"]}, index=[8, 3, 99])
        table = monthly_market(X, np.array([1000, 2000, 9000]))
        np.testing.assert_array_equal(table["median"], [1500, 1500, 9000])
        self.assertEqual(table.iloc[1]["size"], 0)
        self.assertFalse(table.iloc[1]["observed"])
        future_changed = monthly_market(X, [1000, 2000, 99999])
        np.testing.assert_array_equal(table["median"].iloc[:2], future_changed["median"].iloc[:2])

    def test_windows_end_before_target_and_use_no_current_label(self):
        values = np.arange(12, dtype=float) ** 2
        windows, target, positions = monthly_windows(values, 3, 0., 1.)
        for i, t in enumerate(positions):
            np.testing.assert_array_equal(windows[i, :, 0], values[t-3:t])
            self.assertEqual(target[i], values[t] - values[t-1])
        changed = values.copy()
        changed[3] = 99999
        updated, _, _ = monthly_windows(changed, 3, 0., 1.)
        torch.testing.assert_close(windows[0], updated[0])

    def test_clone_unknown_nan_shape_and_inference_invariance_all_models(self):
        raw = fixture()
        fold = get_splits(raw)[0]
        pipeline = FeaturePipeline(["basic", "block"])
        X = pipeline.fit_transform(raw.iloc[fold.train_idx])
        query = pipeline.transform(raw.iloc[fold.valid_idx]).iloc[:11].copy()
        query["TOWN"] = "new town"
        query["YEAR_COMPLETED"] = np.nan
        for kind in ("mlp", "transformer", "carry", "rnn", "lstm", "temporal_transformer"):
            with self.subTest(kind=kind):
                model = tiny(kind).fit(X, raw.iloc[fold.train_idx].MONTHLY_RENT)
                self.assertFalse(hasattr(clone(model), "network_"))
                self.assertFalse(hasattr(clone(model), "property_"))
                prediction = model.predict(query)
                self.assertEqual(prediction.shape, (len(query),))
                self.assertTrue(np.isfinite(prediction).all() and (prediction > 0).all())
                np.testing.assert_allclose(prediction, model.predict(query.iloc[::-1])[::-1], atol=.01, rtol=0)
                batched = np.r_[model.predict(query.iloc[:5]), model.predict(query.iloc[5:])]
                np.testing.assert_allclose(prediction, batched, atol=.01, rtol=0)
                np.testing.assert_array_equal(prediction, model.predict(query.assign(MONTHLY_RENT=1e9, Id=-1)))

    def test_shared_runner_validation_label_perturbation(self):
        raw = fixture()
        changed = raw.copy()
        fold = get_splits(raw)[0]
        changed.iloc[fold.valid_idx, changed.columns.get_loc("MONTHLY_RENT")] += 100000
        for kind in ("mlp", "transformer", "carry", "rnn", "lstm", "temporal_transformer"):
            predictions = []
            for data in (raw, changed):
                with tempfile.TemporaryDirectory() as directory, patch("src.experiment.load_train", return_value=data):
                    result = run_experiment(kind, "test", ["basic", "block"], tiny(kind), output_dir=directory)
                    saved = pd.read_csv(result.run_dir / "validation_predictions.csv")
                    np.testing.assert_array_equal(saved.row_index, fold.valid_idx)
                    predictions.append(saved.Predicted.to_numpy())
            np.testing.assert_array_equal(*predictions)

    def test_identical_seed_reproducible_and_early_stopping_is_internal(self):
        raw = fixture().iloc[:105]
        X = FeaturePipeline(["basic", "block"]).fit_transform(raw)
        for kind in ("mlp", "transformer", "rnn", "lstm", "temporal_transformer"):
            a, b = tiny(kind).fit(X, raw.MONTHLY_RENT), tiny(kind).fit(X, raw.MONTHLY_RENT)
            np.testing.assert_array_equal(a.predict(X.iloc[:8]), b.predict(X.iloc[:8]))
            audit = a.audit_ if kind in ("mlp", "transformer") else a.property_.audit_
            self.assertLess(audit["inner"]["train_end"], audit["inner"]["valid_start"])
            self.assertLessEqual(audit["inner"]["valid_end"], audit["origin"])
            if kind in ("rnn", "lstm"):
                temporal = a.market_.audit_
                self.assertLess(temporal["last_input_month"], temporal["last_target_month"])
                self.assertLess(temporal["inner_train_end"], temporal["inner_valid_start"])

    def test_market_origin_and_maximum_horizon_unchanged_by_query_batch(self):
        raw = fixture().iloc[:105]
        table = monthly_market(raw, raw.MONTHLY_RENT)
        forecaster = MarketForecaster("lstm", epochs=3, threads=1).fit(table)
        months = [str(forecaster.origin_), str(forecaster.origin_+16), str(forecaster.origin_+1)]
        prediction = forecaster.predict_months(months)
        self.assertEqual(prediction[0], table["median"].iloc[-1])
        np.testing.assert_array_equal(prediction, [forecaster.predict_months([m])[0] for m in months])
        self.assertEqual(forecaster.origin_, table.index[-1])

    def test_tuning_recent_window_and_weighting_are_fold_local(self):
        raw = fixture()
        X = FeaturePipeline(["basic", "block"]).fit_transform(raw)
        model = TabularNNRegressor(epochs=2, inner_months=2, batch_size=64,
                                   train_months=6, embedding_size=8, hidden_size=16,
                                   market_normalize=True, loss_weighting="market").fit(X, raw.MONTHLY_RENT)
        self.assertEqual(model.audit_["available_rows"], 153)
        self.assertEqual(model.audit_["train_rows"], 18)
        self.assertEqual(model.audit_["fit_start"], "2024-10")
        self.assertEqual(model.audit_["inner"]["train_end"], "2025-01")
        self.assertEqual(clone(model).embedding_size, 8)
        weights = model._weights(X, raw.MONTHLY_RENT.to_numpy())
        self.assertAlmostEqual(float(weights.mean()), 1., places=6)
        self.assertTrue((weights > 0).all())
        expected = np.repeat((1800 + np.arange(51) * 20.) ** 2, 3)
        np.testing.assert_allclose(weights, expected / expected.mean(), atol=1e-6)
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                TabularNNRegressor(train_months=bad).fit(X, raw.MONTHLY_RENT)

    def test_tuning_hybrid_parameters_clone_and_nonzero_forecast_shrinkage(self):
        model = MarketHybridRegressor(property_dropout=0., embedding_size=16,
                                       train_months=24, loss_weighting="market",
                                       forecast_strength=.25, temporal_dropout=.2,
                                       temporal_weight_decay=.001)
        self.assertEqual(model.get_params(), clone(model).get_params())
        table = monthly_market(fixture(), fixture().MONTHLY_RENT)
        full = MarketForecaster("damped", damping=.5, forecast_strength=1.).fit(table)
        small = MarketForecaster("damped", damping=.5, forecast_strength=.25).fit(table)
        months = pd.period_range(full.origin_+1, full.origin_+16, freq="M").astype(str)
        last = table["median"].iloc[-1]
        np.testing.assert_allclose(small.predict_months(months)-last,
                                   (full.predict_months(months)-last)*.25, atol=1e-9)
        self.assertTrue((small.predict_months(months) != last).all())
        with self.assertRaises(ValueError):
            MarketForecaster("rnn", forecast_strength=0.).fit(table)

    def test_same_family_ensemble_is_exact_mean_and_cloneable(self):
        raw = fixture()
        fold = get_splits(raw)[0]
        pipeline = FeaturePipeline(["basic", "block"])
        train = raw.iloc[fold.train_idx]
        X, query = pipeline.fit_transform(train), pipeline.transform(raw.iloc[fold.valid_idx])
        for kind in ("mlp", "transformer"):
            params = dict(kind=kind, epochs=2, batch_size=64, hidden_size=16,
                          d_model=8, nhead=2, layers=1, inner_months=2,
                          market_normalize=True, random_state=5228, threads=1)
            members = [params, dict(params, dropout=0.)]
            model = NeuralFamilyEnsemble(members).fit(X, train.MONTHLY_RENT)
            self.assertFalse(hasattr(clone(model), "property_"))
            expected = np.mean([m.predict(query) for m in model.property_.models_], axis=0)
            np.testing.assert_allclose(model.predict(query), expected, atol=1e-9, rtol=0)
            np.testing.assert_allclose(model.predict(query), model.predict(query.iloc[::-1])[::-1], atol=.01, rtol=0)
            np.testing.assert_array_equal(model.predict(query), model.predict(query.assign(MONTHLY_RENT=1e9)))

    def test_ensemble_shared_runner_validation_labels_do_not_affect_predictions(self):
        raw = fixture()
        fold = get_splits(raw)[0]
        changed = raw.copy()
        changed.iloc[fold.valid_idx, changed.columns.get_loc("MONTHLY_RENT")] += 100000
        for kind in ("mlp", "transformer"):
            params = dict(kind=kind, epochs=2, batch_size=64, hidden_size=16,
                          d_model=8, nhead=2, layers=1, inner_months=2,
                          market_normalize=True, random_state=5228, threads=1)
            estimator = NeuralFamilyEnsemble([params, dict(params, dropout=0.)])
            predictions = []
            for data in (raw, changed):
                with tempfile.TemporaryDirectory() as directory, patch("src.experiment.load_train", return_value=data):
                    result = run_experiment("ensemble", "test", ["basic", "block"], estimator,
                                            output_dir=directory)
                    saved = pd.read_csv(result.run_dir / "validation_predictions.csv")
                    np.testing.assert_array_equal(saved.row_index, fold.valid_idx)
                    predictions.append(saved.Predicted.to_numpy())
            np.testing.assert_array_equal(*predictions)

    def test_process_parallel_members_reproduce_serial_training(self):
        raw = fixture().iloc[:105]
        X = FeaturePipeline(["basic", "block"]).fit_transform(raw)
        for kind in ("mlp", "transformer"):
            params = dict(kind=kind, epochs=2, batch_size=64, hidden_size=16,
                          d_model=8, nhead=2, layers=1, inner_months=2,
                          market_normalize=True, random_state=5228, threads=1)
            members = [params, dict(params, dropout=0.)]
            serial = NeuralFamilyEnsemble(members, n_jobs=1).fit(X, raw.MONTHLY_RENT)
            parallel = NeuralFamilyEnsemble(members, n_jobs=2).fit(X, raw.MONTHLY_RENT)
            np.testing.assert_array_equal(serial.predict(X), parallel.predict(X))
            self.assertEqual(serial.property_.training_history_, parallel.property_.training_history_)
            self.assertEqual(serial.property_.audit_, parallel.property_.audit_)
            self.assertEqual(clone(parallel).n_jobs, 2)

    def test_temporal_attention_positions_causal_mask_and_zero_increment_start(self):
        network = MarketEncoder("transformer", hidden_size=8, dropout=0.).eval()
        windows = torch.arange(12, dtype=torch.float32).reshape(2, 6, 1)
        torch.testing.assert_close(network(windows), torch.zeros(2), rtol=0, atol=0)
        captured = []
        handle = network.encoder.register_forward_hook(lambda module, args, output: captured.append(output.detach().clone()))
        network(windows)
        changed = windows.clone()
        changed[:, -1] += 1000
        network(changed)
        handle.remove()
        torch.testing.assert_close(captured[0][:, :-1], captured[1][:, :-1], rtol=0, atol=0)
        self.assertFalse(torch.equal(captured[0][:, -1], captured[1][:, -1]))
        with self.assertRaises(ValueError):
            MarketEncoder("transformer", hidden_size=3)

    def test_temporal_transformer_ensemble_future_queries_and_training_locality(self):
        raw = fixture().iloc[:105]
        X = FeaturePipeline(["basic", "block"]).fit_transform(raw)
        member = dict(kind="mlp", epochs=2, hidden_size=16, batch_size=64,
                      market_normalize=True, inner_months=2)
        estimator = NeuralFamilyEnsemble([member], encoder="transformer",
                                         temporal_parameters=dict(epochs=3, hidden_size=8, lookback=6))
        a, b = clone(estimator).fit(X, raw.MONTHLY_RENT), clone(estimator).fit(X, raw.MONTHLY_RENT)
        query = X.iloc[:3].copy()
        query["RENT_APPROVAL_DATE"] = ["2023-11", "2025-03", "2023-12"]
        np.testing.assert_array_equal(a.predict(query), b.predict(query))
        np.testing.assert_allclose(a.predict(query), a.predict(query.iloc[::-1])[::-1], atol=.01, rtol=0)
        np.testing.assert_array_equal(a.predict(query), a.predict(query.assign(MONTHLY_RENT=1e9)))
        self.assertFalse(hasattr(clone(a), "market_"))
        self.assertEqual(a.market_.audit_["attention_mask"], "causal")
        self.assertLess(a.market_.audit_["last_input_month"], a.market_.audit_["last_target_month"])


if __name__ == "__main__":
    unittest.main()
