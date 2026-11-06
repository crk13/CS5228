"""Check that diagnostics preserve row identities and the recent-window fit."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
import delivery
from src.models.base import ModelWrapper
from src.models.trees import RecentWindowRegressor


class DeliveryTests(unittest.TestCase):
    def test_recorder_preserves_predictions_and_marks_actual_training_rows(self):
        X = pd.DataFrame({"RENT_APPROVAL_DATE": ["2020-01", "2023-10", "2023-11"],
                          "FLOOR_AREA_SQM": [60., 70., 80.]}, index=[7, 9, 11])
        y = np.array([99999., 1000., 3000.])
        base = RecentWindowRegressor(ModelWrapper(DummyRegressor(), columns=["FLOOR_AREA_SQM"]), months=24)
        reference = clone(base).fit(X, y).predict(X)
        with tempfile.TemporaryDirectory() as directory, patch.object(delivery, "ROOT", Path(directory)):
            recorder = clone(delivery.TrainingPredictionRecorder(base, "cache")).fit(X, y)
            np.testing.assert_array_equal(recorder.predict(X), reference)
            np.testing.assert_array_equal(recorder.predict(X), [2000., 2000., 2000.])
            recorded = pd.read_csv(Path(directory) / "cache/holdout.csv")
            np.testing.assert_array_equal(recorded.row_index, [7, 9, 11])
            np.testing.assert_array_equal(recorded.used_for_model_fit, [False, True, True])
            np.testing.assert_array_equal(recorded.MONTHLY_RENT, y)
            state = json.loads((Path(directory) / "cache/holdout.json").read_text(encoding="utf-8"))
            self.assertEqual(state["actual_fit_rows"], 2)


if __name__ == "__main__":
    unittest.main()
