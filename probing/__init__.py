"""Intermediate-answer probing for the CIKM 2026 agentic RAG experiments."""

from .agents import ProbedR1Searcher, ProbedSearchR1
from .backend import ProbedHuggingFaceBackend
from .confidence import (
    DEFAULT_TAIL_TOKENS,
    confidence_feature_rows,
    mean_token_logprob,
    probe_confidences,
)
from .run_probing import build_agent, build_confidence_features, run_probing

__all__ = [
    "ProbedSearchR1",
    "ProbedR1Searcher",
    "ProbedHuggingFaceBackend",
    "build_agent",
    "run_probing",
    "build_confidence_features",
    "DEFAULT_TAIL_TOKENS",
    "mean_token_logprob",
    "probe_confidences",
    "confidence_feature_rows",
]
