"""Run intermediate-answer probing or extract probing confidence features.

The CLI exposes two modes:

1. ``run`` executes a probed Search-R1 or R1-Searcher agent over a query CSV,
   saves the trajectory-level probing output, and optionally writes the
   iteration-level ``prob``/``diff_prob`` features used by the paper.
2. ``features`` converts an existing probing-output CSV into the iteration-level
   confidence features without rerunning the agent.

The live probing path uses the paper's canonical E5 dense retrieval setup over a
PyTerrier-DR FlexIndex. In the original experiments the FlexIndex was backed by
an HNSW FAISS index; the index path supplied here should point to that prepared
E5 FlexIndex.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Literal

import pandas as pd
import pyterrier as pt
import pyterrier_dr
from pyterrier_dr import E5

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


def build_e5_retriever(dense_index_path: str | Path) -> pt.Transformer:
    """Build the paper's canonical E5 retriever over a prepared FlexIndex.

    The released experiments used an HNSW-backed E5 FlexIndex. HNSW is therefore
    treated as part of the prepared index rather than exposed as a separate CLI
    option.
    """
    dense_index = pyterrier_dr.FlexIndex(str(dense_index_path))
    query_encoder = E5()
    return query_encoder >> dense_index


def run_probing(
    queries: pd.DataFrame,
    agent: pt.Transformer,
    *,
    batch_size: int = 6,
) -> pd.DataFrame:
    """Run an agent over a query dataframe in deterministic dataframe batches."""
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

    Probe 0 is the zero-retrieval answer obtained after initial reasoning ``r0``
    and before external retrieval. Iteration row 0 uses probe 1 and its change
    from probe 0.
    """
    if "qid" not in probing_results.columns:
        raise ValueError("probing_results is missing required column: qid")

    frames = []
    for row in probing_results.itertuples(index=False):
        if hasattr(row, "probed_probs_after_think"):
            probs = _parse_list(row.probed_probs_after_think)
            if probs is None:
                continue
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


def _parse_json_object(value: str | None, *, name: str) -> dict:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    return parsed


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Saved {len(frame)} rows to {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a probed agent and optionally emit iteration-level features.",
    )
    run_parser.add_argument("--queries-csv", type=Path, required=True)
    run_parser.add_argument(
        "--model",
        choices=["search_r1", "r1_searcher"],
        required=True,
        help="Agent implementation to probe.",
    )
    run_parser.add_argument(
        "--dense-index",
        type=Path,
        required=True,
        help="Path to the prepared HNSW-backed E5 PyTerrier-DR FlexIndex.",
    )
    run_parser.add_argument("--output-csv", type=Path, required=True)
    run_parser.add_argument("--features-output-csv", type=Path, default=None)
    run_parser.add_argument("--top-k", type=int, default=3)
    run_parser.add_argument("--batch-size", type=int, default=6)
    run_parser.add_argument(
        "--hf-model",
        default=None,
        help="Optional Hugging Face model id overriding the agent default.",
    )
    run_parser.add_argument(
        "--backend-args",
        default=None,
        help="Optional JSON object passed as backend_args to the probing wrapper.",
    )
    run_parser.add_argument(
        "--agent-kwargs",
        default=None,
        help="Optional JSON object of additional agent constructor arguments.",
    )

    features_parser = subparsers.add_parser(
        "features",
        help="Extract prob/diff_prob from an existing probing-output CSV.",
    )
    features_parser.add_argument("--probing-csv", type=Path, required=True)
    features_parser.add_argument("--output-csv", type=Path, required=True)

    return parser


def _run_cli(args: argparse.Namespace) -> None:
    if args.command == "features":
        probing_results = pd.read_csv(args.probing_csv)
        features = build_confidence_features(probing_results)
        _write_csv(features, args.output_csv)
        return

    queries = pd.read_csv(args.queries_csv)
    retriever = build_e5_retriever(args.dense_index)

    backend_args = _parse_json_object(args.backend_args, name="backend args") or None
    agent_kwargs = _parse_json_object(args.agent_kwargs, name="agent kwargs")
    if args.hf_model is not None:
        agent_kwargs["model"] = args.hf_model

    agent = build_agent(
        args.model,
        retriever,
        top_k=args.top_k,
        backend_args=backend_args,
        **agent_kwargs,
    )
    probing_results = run_probing(
        queries,
        agent,
        batch_size=args.batch_size,
    )
    _write_csv(probing_results, args.output_csv)

    if args.features_output_csv is not None:
        features = build_confidence_features(probing_results)
        _write_csv(features, args.features_output_csv)


def main() -> None:
    args = _build_parser().parse_args()
    _run_cli(args)


if __name__ == "__main__":
    main()


__all__ = [
    "build_agent",
    "build_e5_retriever",
    "run_probing",
    "build_confidence_features",
]
