UNSUPERVISED_FEATURES = [
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

SUPERVISED_UTILITY_FEATURES = [
    "u_ii_pred",
    "u_ld_pred",
    "u_at_pred",
]

SUPERVISED_QUALITY_FEATURES = [
    "p_ii_pred",
    "p_ld_pred",
    "p_at_pred",
]

# The supervised signal family contains the outputs of both target-specific
# predictor sets.
SUPERVISED_FEATURES = SUPERVISED_UTILITY_FEATURES + SUPERVISED_QUALITY_FEATURES

PROBING_FEATURES = [
    "prob",
    "diff_prob",
]

ITERATION_FEATURES = [
    "sub_qid_feature",
]


def get_prediction_head_input_columns(
    signal_groups=("unsupervised", "supervised", "probing"),
    include_iteration=False,
):
    """Return the feature columns consumed by the prediction head.

    Parameters
    ----------
    signal_groups:
        Any subset of ``unsupervised``, ``supervised`` and ``probing``.
        These names match the signal-family terminology used in Table 3.
    include_iteration:
        Include the optional iteration-index feature. It is disabled by default
        because it is not one of the three signal families reported in Table 3.
    """
    groups = tuple(signal_groups)
    allowed = {"unsupervised", "supervised", "probing"}
    unknown = set(groups) - allowed
    if unknown:
        raise ValueError(f"Unknown signal groups: {sorted(unknown)}")

    columns = []
    if "unsupervised" in groups:
        columns += UNSUPERVISED_FEATURES
    if "supervised" in groups:
        columns += SUPERVISED_FEATURES
    if "probing" in groups:
        columns += PROBING_FEATURES
    if include_iteration:
        columns += ITERATION_FEATURES

    return columns
