"""Bridge prediction-head outputs to the joint early-stopping controller."""

from __future__ import annotations

import pandas as pd

from .joint_controller import apply_joint_controller


_KEY_COLUMNS = ["qid", "original_qid", "sub_qid"]


def merge_prediction_outputs(
    quality_predictions: pd.DataFrame,
    utility_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Merge quality and utility prediction-head outputs safely.

    The two prediction heads must emit exactly one row for every trajectory
    iteration and agree on ``qid``, ``original_qid`` and ``sub_qid``.
    """

    quality_required = set(_KEY_COLUMNS + ["predicted_performance"])
    utility_required = set(_KEY_COLUMNS + ["predicted_utility"])

    missing_quality = quality_required.difference(quality_predictions.columns)
    missing_utility = utility_required.difference(utility_predictions.columns)
    if missing_quality:
        raise ValueError(f"quality predictions are missing columns: {sorted(missing_quality)}")
    if missing_utility:
        raise ValueError(f"utility predictions are missing columns: {sorted(missing_utility)}")

    if quality_predictions.duplicated(_KEY_COLUMNS).any():
        raise ValueError("quality predictions contain duplicate trajectory iterations")
    if utility_predictions.duplicated(_KEY_COLUMNS).any():
        raise ValueError("utility predictions contain duplicate trajectory iterations")

    quality = quality_predictions[_KEY_COLUMNS + ["predicted_performance"]].copy()
    utility = utility_predictions[_KEY_COLUMNS + ["predicted_utility"]].copy()

    merged = quality.merge(
        utility,
        on=_KEY_COLUMNS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not (merged["_merge"] == "both").all():
        unmatched = merged.loc[merged["_merge"] != "both", _KEY_COLUMNS + ["_merge"]]
        raise ValueError(
            "quality and utility predictions do not cover the same trajectory iterations: "
            f"{unmatched.to_dict(orient='records')}"
        )

    merged = merged.drop(columns=["_merge"])
    merged["sub_qid"] = merged["sub_qid"].astype(int)
    merged = merged.sort_values(["original_qid", "sub_qid", "qid"]).reset_index(drop=True)

    # One iteration index must map to exactly one qid inside each trajectory.
    if merged.duplicated(["original_qid", "sub_qid"]).any():
        raise ValueError("multiple qids map to the same trajectory iteration")

    return merged


def apply_controller_to_predictions(
    quality_predictions: pd.DataFrame,
    utility_predictions: pd.DataFrame,
    quality_threshold: float,
    utility_threshold: float,
) -> pd.DataFrame:
    """Apply the joint controller trajectory-by-trajectory.

    Returns one row per ``original_qid`` with both the selected output
    iteration and the iteration at which the stopping decision was made.
    """

    merged = merge_prediction_outputs(quality_predictions, utility_predictions)
    rows = []

    for original_qid, trajectory in merged.groupby("original_qid", sort=False):
        trajectory = trajectory.sort_values("sub_qid").reset_index(drop=True)
        decision = apply_joint_controller(
            predicted_quality=trajectory["predicted_performance"].tolist(),
            predicted_utility=trajectory["predicted_utility"].tolist(),
            quality_threshold=quality_threshold,
            utility_threshold=utility_threshold,
        )

        output_row = trajectory.iloc[decision.output_index]
        decision_row = trajectory.iloc[decision.decision_index]
        rows.append(
            {
                "original_qid": original_qid,
                "output_qid": output_row["qid"],
                "output_sub_qid": int(output_row["sub_qid"]),
                "decision_qid": decision_row["qid"],
                "decision_sub_qid": int(decision_row["sub_qid"]),
                "reason": decision.reason,
                "states": decision.states,
            }
        )

    return pd.DataFrame(rows)
