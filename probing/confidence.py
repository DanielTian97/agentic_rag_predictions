"""Confidence features derived from intermediate-answer probe generations."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_TAIL_TOKENS = 4


def mean_token_logprob(
    token_logprobs: Sequence[float] | None,
    *,
    tail_tokens: int = DEFAULT_TAIL_TOKENS,
) -> float:
    """Return the length-normalized log-probability used by the paper.

    The final four generated-token scores are discarded before averaging; these
    tokens correspond to the forced-answer closing tail. The same convention is
    used for both agents.
    """

    if token_logprobs is None:
        return float("nan")

    values = np.asarray(list(token_logprobs), dtype=float)
    if tail_tokens < 0:
        raise ValueError("tail_tokens must be non-negative")
    if tail_tokens:
        if len(values) <= tail_tokens:
            return float("nan")
        values = values[:-tail_tokens]

    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def probe_confidences(
    probe_token_logprobs: Iterable[Sequence[float] | None],
    *,
    tail_tokens: int = DEFAULT_TAIL_TOKENS,
) -> list[float]:
    """Reduce token-level probe log-probabilities to one scalar per probe."""

    return [
        mean_token_logprob(logprobs, tail_tokens=tail_tokens)
        for logprobs in probe_token_logprobs
    ]


def confidence_feature_rows(
    qid: str,
    probe_token_logprobs: Iterable[Sequence[float] | None],
    *,
    tail_tokens: int = DEFAULT_TAIL_TOKENS,
) -> pd.DataFrame:
    """Build the paper's ``prob`` and ``diff_prob`` features for one trajectory.

    Probe 0 is the zero-retrieval (zero-shot) answer obtained after the model's
    initial reasoning ``r0`` and before any external retrieval. Each subsequent
    probe corresponds to one retrieval-reasoning iteration, so zero-based
    dataframe row ``i`` uses:

    ``prob_i = confidence(probe_{i+1})``
    ``diff_prob_i = confidence(probe_{i+1}) - confidence(probe_i)``

    The first ``diff_prob`` is therefore *not* zero; it measures the change from
    the zero-retrieval answer after initial reasoning to the first retrieval
    iteration.
    """

    confidences = probe_confidences(
        probe_token_logprobs,
        tail_tokens=tail_tokens,
    )
    if len(confidences) < 2:
        return pd.DataFrame(columns=["qid", "sub_qid", "prob", "diff_prob", "zero_prob"])

    zero_prob = confidences[0]
    rows = []
    for i in range(len(confidences) - 1):
        current = confidences[i + 1]
        previous = confidences[i]
        rows.append(
            {
                "qid": f"{qid}-{i}",
                "sub_qid": i,
                "prob": current,
                "diff_prob": current - previous,
                "zero_prob": zero_prob,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_TAIL_TOKENS",
    "mean_token_logprob",
    "probe_confidences",
    "confidence_feature_rows",
]
