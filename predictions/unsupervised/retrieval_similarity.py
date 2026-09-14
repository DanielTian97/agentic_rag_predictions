"""Retrieval-list similarity features used by the CIKM 2026 predictors.

The public release consumes the trajectory and retrieval outputs as CSV dataframes.
This module replaces the experiment-specific scripts that wrote JSONL / pickle
artifacts and computes the final RBO features directly in tabular form.
"""
from __future__ import annotations

import pandas as pd
import pyterrier_alpha as pta


REQUIRED_RETRIEVAL_COLUMNS = {"qid", "rank"}


def _prepare_retrieval(retrieval: pd.DataFrame) -> pd.DataFrame:
    """Normalize a per-iteration retrieval dataframe.

    Expected iteration qids have the form ``<original-qid>-<iteration>``.  We
    split from the right so original dataset qids may themselves contain '-'.
    """
    missing = REQUIRED_RETRIEVAL_COLUMNS - set(retrieval.columns)
    if missing:
        raise ValueError(f"Retrieval dataframe is missing columns: {sorted(missing)}")

    frame = retrieval.copy()
    frame = frame[frame["qid"].astype(str) != "qid"].copy()
    frame["qid"] = frame["qid"].astype(str)
    split = frame["qid"].str.rsplit("-", n=1, expand=True)
    if split.shape[1] != 2:
        raise ValueError("Iteration retrieval qids must end in '-<iteration>'.")
    frame["original_qid"] = split[0]
    frame["sub_qid"] = pd.to_numeric(split[1], errors="raise").astype(int)
    frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").astype(int)
    if "score" in frame.columns:
        frame["score"] = pd.to_numeric(frame["score"], errors="raise")
    return frame


def _rbo(left: pd.DataFrame, right: pd.DataFrame, cutoff: int) -> float:
    """Return PyTerrier Alpha's RBO for two ranked lists."""
    left = left[left["rank"] < cutoff].copy()
    right = right[right["rank"] < cutoff].copy()
    if left.empty or right.empty:
        return 0.0

    # pta.rbo groups by qid, so give both lists the same temporary qid.
    left["qid"] = "q"
    right["qid"] = "q"
    result = list(pta.rbo(left, right))
    return float(result[0][1]) if result else 0.0


def compute_inter_iteration_rbo(
    retrieval: pd.DataFrame,
    cutoff: int,
) -> pd.DataFrame:
    """Compute maximum retrieval-list RBO with any previous iteration.

    This reproduces the final ``max_int_iter_rbo_*`` signal used by the
    sequential predictor.  Iteration 0 has no predecessor and therefore gets
    a value of 0, matching the original experiment code.
    """
    frame = _prepare_retrieval(retrieval)
    records: list[dict[str, object]] = []

    for original_qid, group in frame.groupby("original_qid", sort=False):
        iterations = sorted(group["sub_qid"].unique())
        previous: list[pd.DataFrame] = []
        for sub_qid in iterations:
            current = group[group["sub_qid"] == sub_qid]
            similarities = [_rbo(current, old, cutoff) for old in previous]
            records.append(
                {
                    "qid": f"{original_qid}-{sub_qid}",
                    f"max_int_iter_rbo_{cutoff}": max(similarities) if similarities else 0.0,
                }
            )
            previous.append(current)

    return pd.DataFrame(records)


def compute_question_iteration_rbo(
    retrieval: pd.DataFrame,
    original_query_retrieval: pd.DataFrame,
    cutoff: int,
) -> pd.DataFrame:
    """Compute RBO between original-question and current-iteration retrievals.

    ``original_query_retrieval`` is the precomputed retrieval run for the
    original questions.  Its qids are the unmodified original qids.
    """
    frame = _prepare_retrieval(retrieval)
    original = original_query_retrieval.copy()
    missing = REQUIRED_RETRIEVAL_COLUMNS - set(original.columns)
    if missing:
        raise ValueError(
            "Original-query retrieval dataframe is missing columns: "
            f"{sorted(missing)}"
        )
    original["qid"] = original["qid"].astype(str)
    original["rank"] = pd.to_numeric(original["rank"], errors="raise").astype(int)

    records: list[dict[str, object]] = []
    for (original_qid, sub_qid), current in frame.groupby(
        ["original_qid", "sub_qid"], sort=False
    ):
        baseline = original[original["qid"] == str(original_qid)]
        value = _rbo(baseline, current, cutoff)
        records.append(
            {
                "qid": f"{original_qid}-{sub_qid}",
                f"topic_drift_rbo_{cutoff}": value,
            }
        )

    return pd.DataFrame(records)


def compute_retrieval_similarity_features(
    retrieval: pd.DataFrame,
    original_query_retrieval: pd.DataFrame,
    k: int = 3,
    broad_cutoff: int = 20,
) -> pd.DataFrame:
    """Return the four RBO columns used by the final CIKM predictor."""
    frames = [
        compute_inter_iteration_rbo(retrieval, k),
        compute_inter_iteration_rbo(retrieval, broad_cutoff),
        compute_question_iteration_rbo(retrieval, original_query_retrieval, k),
        compute_question_iteration_rbo(
            retrieval, original_query_retrieval, broad_cutoff
        ),
    ]

    result = frames[0]
    for other in frames[1:]:
        result = result.merge(other, on="qid", how="outer", validate="one_to_one")
    return result.sort_values("qid").reset_index(drop=True)
