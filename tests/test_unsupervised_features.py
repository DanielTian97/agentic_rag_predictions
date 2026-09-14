import unittest

import pandas as pd

from predictions.unsupervised.features import (
    UNSUPERVISED_FEATURE_COLUMNS,
    merge_unsupervised_features,
)


class UnsupervisedFeatureAssemblyTest(unittest.TestCase):
    def test_merge_final_feature_block(self):
        qpp = pd.DataFrame(
            {
                "qid": ["q-0"],
                "nqc_score": [1.0],
                "a_pair_ratio_score": [2.0],
                "dense_qpp_score": [3.0],
            }
        )
        rbo = pd.DataFrame(
            {
                "qid": ["q-0"],
                "max_int_iter_rbo_k": [0.0],
                "max_int_iter_rbo_20": [0.0],
                "topic_drift_rbo_k": [0.4],
                "topic_drift_rbo_20": [0.5],
            }
        )
        semantic = pd.DataFrame(
            {
                "qid": ["q-0"],
                "think_sim": [0.8],
                "topic_drift_sim": [0.7],
            }
        )

        out = merge_unsupervised_features(qpp, rbo, semantic)
        self.assertEqual(out.columns.tolist(), ["qid", *UNSUPERVISED_FEATURE_COLUMNS])
        self.assertEqual(out.loc[0, "think_sim"], 0.8)

    def test_rejects_incomplete_feature_block(self):
        with self.assertRaises(ValueError):
            merge_unsupervised_features(pd.DataFrame({"qid": ["q-0"], "think_sim": [0.1]}))


if __name__ == "__main__":
    unittest.main()
