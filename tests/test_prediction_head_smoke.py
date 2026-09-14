import unittest

import numpy as np
import pandas as pd

from predictions.prediction_head.features import get_prediction_head_input_columns
from predictions.prediction_head.mlp import build_prediction_head
from predictions.prediction_head.train import (
    fit_prediction_head,
    prepare_prediction_head_data,
)


class PredictionHeadSmokeTest(unittest.TestCase):
    @staticmethod
    def _tiny_dataframe():
        feature_columns = get_prediction_head_input_columns()
        rows = []

        for split, n_questions in (("train", 100), ("test", 10)):
            for q in range(n_questions):
                original_qid = f"{split}-q{q}"
                for iteration in range(3):
                    base = (q + 1) / (n_questions + 1) + 0.05 * iteration
                    row = {
                        "qid": f"{original_qid}-{iteration}",
                        "original_qid": original_qid,
                        "sub_qid": iteration,
                        "split": split,
                        "performance": 0.25 + 0.45 * base,
                        "utility": 0.08 - 0.02 * iteration + 0.01 * base,
                    }
                    for j, column in enumerate(feature_columns):
                        row[column] = base + 0.01 * j
                    rows.append(row)

        return pd.DataFrame(rows)

    def test_camera_ready_mlp_shape(self):
        model = build_prediction_head(random_state=0)
        self.assertEqual(model.hidden_layer_sizes, (16, 8))

    def test_window_construction_runs_for_w1_and_w3(self):
        dataframe = self._tiny_dataframe()
        n_signals = len(get_prediction_head_input_columns())

        for window_size in (1, 3):
            seq_df, columns = prepare_prediction_head_data(
                dataframe,
                target="performance",
                window_size=window_size,
                include_type=False,
            )
            self.assertEqual(len(seq_df), len(dataframe))
            self.assertEqual(len(columns), n_signals * window_size)
            self.assertTrue(set(columns).issubset(seq_df.columns))

    def test_end_to_end_fit_and_predict_continuous_targets(self):
        dataframe = self._tiny_dataframe()

        for target in ("performance", "utility"):
            model, predictions, (y_test, y_pred), output_path = fit_prediction_head(
                dataframe,
                target=target,
                window_size=3,
                include_type=False,
                random_state=0,
            )

            self.assertEqual(len(predictions), 30)
            self.assertEqual(len(y_test), 30)
            self.assertEqual(len(y_pred), 30)
            self.assertIn(f"predicted_{target}", predictions.columns)
            self.assertTrue(np.isfinite(y_pred).all())
            self.assertTrue(np.issubdtype(y_pred.dtype, np.floating))
            self.assertIsNotNone(model.model)
            self.assertIsNone(output_path)


if __name__ == "__main__":
    unittest.main()
