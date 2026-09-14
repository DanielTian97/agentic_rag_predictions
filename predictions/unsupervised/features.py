"""Assembly of the nine training-free features used by the prediction head."""

from __future__ import annotations

from functools import reduce

import pandas as pd


UNSUPERVISED_FEATURE_COLUMNS = [
    "max_int_iter_rbo_k",
    "max_int_iter_rbo_20",
    "think_sim",
    "topic_drift_rbo_20",
    "topic_drift_rbo_k",
    "topic_drift_sim",
    "nqc_score",
    "a_pair_ratio_score",
    "dense_qpp_score",
]


def merge_unsupervised_features(*frames: pd.DataFrame) -> pd.DataFrame:
    """Merge feature dataframes on iteration-level ``qid``.

    Each input must contain one row per iteration qid. Extra provenance columns
    such as ``query`` or ``orig_qid`` are ignored unless they are the only copy
    available; the returned dataframe contains ``qid`` plus the nine
    unsupervised feature columns in prediction-head order.
    """
    if not frames:
        return pd.DataFrame(columns=["qid", *UNSUPERVISED_FEATURE_COLUMNS])

    selected = []
    for frame in frames:
        if "qid" not in frame.columns:
            raise ValueError("Every feature dataframe must contain a 'qid' column.")
        if frame["qid"].duplicated().any():
            duplicates = frame.loc[frame["qid"].duplicated(), "qid"].astype(str).tolist()
            raise ValueError(f"Feature dataframe contains duplicate qids: {duplicates[:5]}")

        keep = ["qid"] + [
            c for c in UNSUPERVISED_FEATURE_COLUMNS if c in frame.columns
        ]
        selected.append(frame[keep].copy())

    merged = reduce(
        lambda left, right: left.merge(
            right,
            on="qid",
            how="outer",
            validate="one_to_one",
        ),
        selected,
    )

    missing = [c for c in UNSUPERVISED_FEATURE_COLUMNS if c not in merged.columns]
    if missing:
        raise ValueError(
            "Incomplete unsupervised feature block; missing columns: "
            f"{missing}"
        )

    return merged[["qid", *UNSUPERVISED_FEATURE_COLUMNS]].sort_values("qid").reset_index(drop=True)
