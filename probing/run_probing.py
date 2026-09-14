"""Small dataframe-first runner for batched intermediate-answer probing."""
from __future__ import annotations

import ast
from typing import Literal

import pandas as pd
import pyterrier as pt

from .agents import ProbedR1Searcher, ProbedSearchR1
from .confidence import confidence_feature_rows


def build_agent(
    model: Literal["search_r1", "r1_searcher"],
    retriever: pt.Transformer,
    *,
    top_k: int = 3,
    backend_args: dict | None = None,
    **agent_kwargs,
):
    """Construct the paper's Hugging Face probing wrapper for one agent."""
    if model == "search_r1":
        return ProbedSearchR1.from_hf(
            retriever,
            top_k=top_k,
            backend_args=backend_args,
            **agent_kwargs,
        )
    if model == "r1_searcher":
        return ProbedR1Searcher.from_hf(
            retriever,
            top_k=top_k,
            backend_args=backend_args,
            **agent_kwargs,
        )
    raise ValueError(f"Unknown model: {model}")


def run_probing(
    queries: pd.DataFrame,
    agent: pt.Transformer,
    *,
    batch_size: int = 6,
) -> pd.DataFrame:
    """Run an agent over a query dataframe in deterministic dataframe batches.

    Parameters
    ----------
    queries:
        Dataframe containing at least ``qid`` and ``query``.
    agent:
        A probed Search-R1 or R1-Searcher transformer.
    batch_size:
        Number of trajectories supplied to the batched agent at a time.
    """
    required = {"qid", "query"}
    missing = required.difference(queries.columns)
    if missing:
        raise ValueError(f"queries is missing required columns: {sorted(missing)}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    outputs = []
    for start in range(0, len(queries), batch_size):
        batch = queries.iloc[start : start + batch_size].copy()
        outputs.append(agent.transform(batch))

    if not outputs:
        return agent.transform(queries.iloc[:0].copy())
    return pd.concat(outputs, ignore_index=True)


def _parse_list(value):
    """Parse list-like values after a probing CSV round-trip."""
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def build_confidence_features(probing_results: pd.DataFrame) -> pd.DataFrame:
    """Expand trajectory-level probe outputs into iteration-level confidence features.

    If ``probed_probs_after_think`` is available, those scalar confidences are
    used directly. Otherwise, ``probed_logits_after_think`` is reduced with the
    mean-token-logprob rule.

    Probe 0 is the zero-retrieval (zero-shot) answer obtained after the model's
    initial reasoning ``r0`` and before any external retrieval. Iteration row 0
    then uses probe 1 and its change from probe 0.
    """
    if "qid" not in probing_results.columns:
        raise ValueError("probing_results is missing required column: qid")

    frames = []
    for row in probing_results.itertuples(index=False):
        if hasattr(row, "probed_probs_after_think"):
            probs = _parse_list(row.probed_probs_after_think)
            if probs is None:
                continue
            # confidence_feature_rows expects token-level traces. Scalar probe
            # confidences are already reduced, so construct rows directly.
            probs = [float(value) for value in probs]
            if len(probs) < 2:
                continue
            zero_prob = probs[0]
            frames.append(
                pd.DataFrame(
                    {
                        "qid": [f"{row.qid}-{i}" for i in range(len(probs) - 1)],
                        "sub_qid": list(range(len(probs) - 1)),
                        "prob": probs[1:],
                        "diff_prob": [
                            probs[i + 1] - probs[i]
                            for i in range(len(probs) - 1)
                        ],
                        "zero_prob": [zero_prob] * (len(probs) - 1),
                    }
                )
            )
            continue

        if hasattr(row, "probed_logits_after_think"):
            raw = _parse_list(row.probed_logits_after_think)
            if raw is not None:
                frames.append(confidence_feature_rows(str(row.qid), raw))

    if not frames:
        return pd.DataFrame(columns=["qid", "sub_qid", "prob", "diff_prob", "zero_prob"])
    return pd.concat(frames, ignore_index=True)


__all__ = ["build_agent", "run_probing", "build_confidence_features"]
