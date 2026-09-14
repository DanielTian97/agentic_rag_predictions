"""Semantic-similarity features used by the CIKM 2026 prediction pipeline.

The original experiments produced these values with separate analysis scripts and
pickled dictionaries. This module keeps the same feature definitions but works
from the CSV/dataframe representation used throughout the CIKM codebase.

Implemented signals
-------------------
``topic_drift_sim``
    Cosine similarity between the original question and each intermediate
    retrieval query.

``think_sim``
    Cosine similarity between adjacent reasoning states, i.e.
    r_{i-1} and r_i.

No r_0-vs-r_i similarity is added here because it was not part of the final
prediction-head feature set used for the reported experiments.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


ArrayLikeEncoder = Callable[[Sequence[str]], np.ndarray]


def _safe_sequence(value) -> list:
    """Parse a list-like CSV cell without using ``eval``."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Expected a list-like value, got {type(parsed).__name__}")
    return list(parsed)


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity for equally shaped 2-D arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError("Expected two equally shaped 2-D embedding arrays")

    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.zeros(len(a), dtype=float)
    valid = denom > 0
    out[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return out


def query_drift_similarity(
    original_question: str,
    intermediate_queries: Sequence[str],
    encode: ArrayLikeEncoder,
) -> list[float]:
    """Return sim(original question, q_i) for every intermediate query."""
    queries = [str(q) for q in intermediate_queries]
    if not queries:
        return []

    original = np.asarray(encode([str(original_question)]), dtype=float)
    query_embs = np.asarray(encode(queries), dtype=float)

    repeated_original = np.repeat(original, len(queries), axis=0)
    return _cosine_rows(repeated_original, query_embs).tolist()


def adjacent_reasoning_similarity(
    reasoning_states: Sequence[str],
    encode: ArrayLikeEncoder,
) -> list[float]:
    """Return sim(r_{i-1}, r_i) for adjacent reasoning states."""
    states = [str(r) for r in reasoning_states]
    if len(states) < 2:
        return []

    embeddings = np.asarray(encode(states), dtype=float)
    return _cosine_rows(embeddings[:-1], embeddings[1:]).tolist()


def _extract_search_r1_reasoning(output: str, all_queries) -> list[str]:
    """Recover Search-R1 reasoning states using the paper experiment logic.

    Search-R1 emits a new ``<think>...</think>`` block after each returned
    ``<information>`` block.  The historical script split the trajectory at the
    closing search token corresponding to each generated query and retained the
    last reasoning block in each resulting segment.
    """
    queries = _safe_sequence(all_queries)
    search_closers = []
    for item in queries:
        query_text = item[1] if isinstance(item, (list, tuple)) and len(item) >= 2 else item
        search_closers.append(f"{query_text}</search>")

    if search_closers:
        parts = re.split("|".join(map(re.escape, search_closers)), str(output))
    else:
        parts = [str(output)]

    kept_parts = []
    for i, part in enumerate(parts):
        if not part.strip():
            continue
        if i == 0 or re.search(r"<information>.*?</information>", part, flags=re.DOTALL):
            kept_parts.append(part.split("</information>")[-1])

    reasoning = []
    for part in kept_parts:
        matches = re.findall(r"<think>(.*?)</think>", part, flags=re.DOTALL)
        reasoning.append(matches[-1] if matches else "")

    expected = len(queries) + 1
    if len(reasoning) != expected:
        raise ValueError(
            f"Expected {expected} Search-R1 reasoning states, found {len(reasoning)}"
        )
    return reasoning


def _extract_r1_searcher_reasoning(output: str) -> list[str]:
    """Recover R1-Searcher reasoning segments around intermediate queries.

    R1-Searcher places ``<|begin_of_query|>...<|end_of_query|>`` markers inside
    a single reasoning trace. Splitting on those query spans gives the initial
    reasoning state followed by one state per retrieval iteration, matching the
    historical feature-extraction script.
    """
    text = str(output)
    # The old implementation analysed only the reasoning portion before the
    # final answer.  When a closing think tag is present, discard the suffix.
    if "</think>" in text:
        text = text.split("</think>", 1)[0]
    text = text.strip()

    query_pattern = r"<\|begin_of_query\|>.*?<\|end_of_query\|>"
    reasoning = [
        segment.strip()
        for segment in re.split(query_pattern, text, flags=re.DOTALL)
    ]
    return reasoning


def extract_reasoning_states(
    output: str,
    *,
    rag_model: str,
    all_queries=None,
) -> list[str]:
    """Extract the reasoning sequence used for ``think_sim``.

    Parameters
    ----------
    output:
        Raw trajectory text stored in the generation CSV.
    rag_model:
        ``"r1"`` for Search-R1 or ``"r1s"`` for R1-Searcher.
    all_queries:
        Search-R1's CSV list of generated intermediate queries. Required for
        ``rag_model="r1"``.
    """
    if rag_model == "r1":
        if all_queries is None:
            raise ValueError("Search-R1 reasoning extraction requires all_queries")
        return _extract_search_r1_reasoning(output, all_queries)
    if rag_model == "r1s":
        return _extract_r1_searcher_reasoning(output)
    raise ValueError("rag_model must be 'r1' or 'r1s'")


def add_query_drift_feature(
    rows: pd.DataFrame,
    *,
    encode: ArrayLikeEncoder,
    query_column: str = "query",
    intermediate_query_column: str = "all_queries",
    qid_column: str = "qid",
) -> pd.DataFrame:
    """Expand trajectory rows into one ``topic_drift_sim`` row per iteration.

    ``all_queries`` may be either a Python list or the string representation of
    a list as stored in the experiment CSV files.
    """
    records = []

    for row in rows.itertuples(index=False):
        original_qid = str(getattr(row, qid_column))
        original_question = getattr(row, query_column)
        queries = _safe_sequence(getattr(row, intermediate_query_column))

        # Search-R1 historically stores (sub_qid, query_text) pairs, whereas
        # R1-Searcher may store query strings. Preserve both formats.
        sub_ids = []
        query_texts = []
        for i, item in enumerate(queries):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                sub_ids.append(int(item[0]))
                query_texts.append(str(item[1]))
            else:
                sub_ids.append(i)
                query_texts.append(str(item))

        sims = query_drift_similarity(original_question, query_texts, encode)
        records.extend(
            {
                "qid": f"{original_qid}-{sub_id}",
                "orig_qid": original_qid,
                "sub_qid": sub_id,
                "topic_drift_sim": sim,
            }
            for sub_id, sim in zip(sub_ids, sims)
        )

    return pd.DataFrame.from_records(
        records,
        columns=["qid", "orig_qid", "sub_qid", "topic_drift_sim"],
    )


def add_adjacent_reasoning_feature(
    trajectories: Iterable[tuple[str, Sequence[str]]],
    *,
    encode: ArrayLikeEncoder,
) -> pd.DataFrame:
    """Create ``think_sim`` rows from pre-extracted reasoning trajectories."""
    records = []

    for original_qid, reasoning_states in trajectories:
        sims = adjacent_reasoning_similarity(reasoning_states, encode)
        records.extend(
            {
                "qid": f"{original_qid}-{i}",
                "orig_qid": str(original_qid),
                "sub_qid": i,
                "think_sim": sim,
            }
            for i, sim in enumerate(sims)
        )

    return pd.DataFrame.from_records(
        records,
        columns=["qid", "orig_qid", "sub_qid", "think_sim"],
    )


def add_adjacent_reasoning_feature_from_csv(
    rows: pd.DataFrame,
    *,
    encode: ArrayLikeEncoder,
    rag_model: str,
    output_column: str = "output",
    query_history_column: str = "all_queries",
    qid_column: str = "qid",
) -> pd.DataFrame:
    """Compute ``think_sim`` directly from generation CSV rows."""
    trajectories = []

    for row in rows.itertuples(index=False):
        qid = str(getattr(row, qid_column))
        output = getattr(row, output_column)
        all_queries = (
            getattr(row, query_history_column)
            if rag_model == "r1"
            else None
        )
        states = extract_reasoning_states(
            output,
            rag_model=rag_model,
            all_queries=all_queries,
        )
        trajectories.append((qid, states))

    return add_adjacent_reasoning_feature(trajectories, encode=encode)


@dataclass
class SentenceTransformerEncoder:
    """Small adapter around ``SentenceTransformer.encode``.

    The dependency is imported lazily so unit tests for the dataframe logic do
    not require downloading a model.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts)), dtype=float)
