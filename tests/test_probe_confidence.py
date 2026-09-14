import math
import unittest

import pandas as pd

from probing.confidence import confidence_feature_rows, mean_token_logprob
from probing.run_probing import build_confidence_features


class ProbeConfidenceTest(unittest.TestCase):
    def test_mean_token_logprob_excludes_historical_tail(self):
        value = mean_token_logprob([1, 2, 3, 4, 5, 6])
        self.assertEqual(value, 1.5)

    def test_confidence_rows_use_zero_probe_as_first_baseline(self):
        rows = confidence_feature_rows(
            "q1",
            [[-3.0, -3.0], [-1.0, -1.0], [-0.5, -0.5]],
            tail_tokens=0,
        )
        self.assertEqual(rows.qid.tolist(), ["q1-0", "q1-1"])
        self.assertEqual(rows.sub_qid.tolist(), [0, 1])
        self.assertEqual(rows.prob.tolist(), [-1.0, -0.5])
        self.assertEqual(rows.diff_prob.tolist(), [2.0, 0.5])
        self.assertEqual(rows.zero_prob.tolist(), [-3.0, -3.0])

    def test_short_probe_is_nan_when_tail_consumes_all_tokens(self):
        self.assertTrue(math.isnan(mean_token_logprob([-1.0, -2.0])))

    def test_csv_scalar_confidences_expand(self):
        trajectories = pd.DataFrame(
            {
                "qid": ["q1"],
                "probed_probs_after_think": ["[-3.0, -1.0, -0.5]"],
            }
        )
        rows = build_confidence_features(trajectories)
        self.assertEqual(rows.qid.tolist(), ["q1-0", "q1-1"])
        self.assertEqual(rows.prob.tolist(), [-1.0, -0.5])
        self.assertEqual(rows.diff_prob.tolist(), [2.0, 0.5])


if __name__ == "__main__":
    unittest.main()
