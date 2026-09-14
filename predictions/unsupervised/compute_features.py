"""Compute the CIKM 2026 unsupervised feature block from CSV files.

Example
-------
python -m predictions.unsupervised.compute_features \
    --generation-csv data/probed_trajectories/hotpotqa_dev.csv \
    --retrieval-csv data/retrieval_results/hotpotqa_dev.csv \
    --original-retrieval-csv data/retrieval_results/hotpotqa_dev_original_query.csv \
    --rag-model r1 \
    --sparse-index path/to/ragwiki_sparse_index \
    --dense-index path/to/ragwiki_e5_index \
    --output-csv data/evaluation_results/hotpotqa_dev_unsupervised.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features import merge_unsupervised_features
from .qpp.calculator import QPPCalculator
from .retrieval_similarity import compute_retrieval_similarity_features
from .semantic_similarity import (
    SentenceTransformerEncoder,
    add_adjacent_reasoning_feature_from_csv,
    add_query_drift_feature,
)


def _normalise_generation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise the two agent implementations to the common CSV schema."""
    out = frame.copy()
    rename = {}
    if "search_history" in out.columns and "all_queries" not in out.columns:
        rename["search_history"] = "all_queries"
    if "search_iterations" in out.columns and "iteration" not in out.columns:
        rename["search_iterations"] = "iteration"
    if rename:
        out = out.rename(columns=rename)
    return out


def _load_qpp_resources(sparse_index_path: str, dense_index_path: str):
    """Load the PyTerrier resources used by the experiments."""
    import pyterrier as pt
    import pyterrier_dr
    from pyterrier_dr import E5

    sparse_index = pt.terrier.TerrierIndex(sparse_index_path)
    dense_index = pyterrier_dr.FlexIndex(dense_index_path)
    query_encoder = E5()
    return sparse_index, dense_index, query_encoder


def compute_unsupervised_features(
    generation: pd.DataFrame,
    retrieval: pd.DataFrame,
    original_query_retrieval: pd.DataFrame,
    *,
    rag_model: str,
    sparse_index,
    dense_index,
    query_encoder,
    k: int = 3,
    broad_cutoff: int = 20,
    semantic_encoder=None,
) -> pd.DataFrame:
    """Compute the nine training-free features used by the prediction head."""
    generation = _normalise_generation_columns(generation)

    required_generation = {"qid", "query", "output", "all_queries"}
    missing = required_generation - set(generation.columns)
    if missing:
        raise ValueError(
            f"Generation dataframe is missing columns: {sorted(missing)}"
        )

    if semantic_encoder is None:
        semantic_encoder = SentenceTransformerEncoder()

    qpp = QPPCalculator(
        sparse_index=sparse_index,
        dense_index=dense_index,
        query_encoder=query_encoder,
    ).compute(retrieval, k=k)

    retrieval_similarity = compute_retrieval_similarity_features(
        retrieval,
        original_query_retrieval,
        k=k,
        broad_cutoff=broad_cutoff,
    )

    query_drift = add_query_drift_feature(
        generation,
        encode=semantic_encoder,
        intermediate_query_column="all_queries",
    )
    think_similarity = add_adjacent_reasoning_feature_from_csv(
        generation,
        encode=semantic_encoder,
        rag_model=rag_model,
        query_history_column="all_queries",
    )

    semantic = query_drift[["qid", "topic_drift_sim"]].merge(
        think_similarity[["qid", "think_sim"]],
        on="qid",
        how="outer",
        validate="one_to_one",
    )

    return merge_unsupervised_features(
        qpp,
        retrieval_similarity,
        semantic,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the CIKM 2026 unsupervised predictor features."
    )
    parser.add_argument("--generation-csv", required=True)
    parser.add_argument("--retrieval-csv", required=True)
    parser.add_argument("--original-retrieval-csv", required=True)
    parser.add_argument("--rag-model", required=True, choices=["r1", "r1s"])
    parser.add_argument("--sparse-index", required=True)
    parser.add_argument("--dense-index", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--broad-cutoff", type=int, default=20)
    parser.add_argument(
        "--semantic-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    generation = pd.read_csv(args.generation_csv)
    retrieval = pd.read_csv(args.retrieval_csv)
    original_retrieval = pd.read_csv(args.original_retrieval_csv)

    sparse_index, dense_index, query_encoder = _load_qpp_resources(
        args.sparse_index,
        args.dense_index,
    )
    semantic_encoder = SentenceTransformerEncoder(args.semantic_model)

    features = compute_unsupervised_features(
        generation,
        retrieval,
        original_retrieval,
        rag_model=args.rag_model,
        sparse_index=sparse_index,
        dense_index=dense_index,
        query_encoder=query_encoder,
        k=args.k,
        broad_cutoff=args.broad_cutoff,
        semantic_encoder=semantic_encoder,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"Saved {len(features)} iteration rows to {output_path}")


if __name__ == "__main__":
    main()
