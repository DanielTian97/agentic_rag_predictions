"""QPP signals for the CIKM 2026 agentic-RAG prediction experiments.

This module follows the cleaned QPP structure used in the ECIR 2026 release,
while preserving the three CIKM signals and their historical formulas:

- ``nqc_score``: NQC from retrieval-score variance and maximum query IDF;
- ``a_pair_ratio_score``: pairwise document-coherence ratio;
- ``dense_qpp_score``: dense semantic-concentration/spatial QPP.

The output column names intentionally match the downstream sequential-prediction
feature pipeline used for the paper.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


class QPPCalculator:
    """Compute the three intra-iteration QPP signals used in the paper."""

    def __init__(self, sparse_index=None, dense_index=None, query_encoder=None):
        self.sparse_index = sparse_index
        self.dense_index = dense_index
        self.query_encoder = query_encoder

    def _terrier_index_obj(self):
        if self.sparse_index is None:
            raise ValueError("NQC requires a PyTerrier sparse index.")
        if hasattr(self.sparse_index, "index_obj"):
            return self.sparse_index.index_obj()
        return self.sparse_index

    def _max_idf(self, query: str) -> float:
        """Historical CIKM NQC normaliser: max log(N / df) over query terms."""

        import pyterrier as pt

        index = self._terrier_index_obj()
        stats = index.getCollectionStatistics()
        doc_num = float(stats.getNumberOfDocuments())
        lexicon = index.getLexicon()
        stemmer = pt.TerrierStemmer.porter

        dfs = []
        for token in str(query).split():
            term = stemmer.stem(token)
            try:
                dfs.append(float(lexicon[term].getDocumentFrequency()))
            except Exception:
                dfs.append(1.0)

        if not dfs:
            return float(np.log(doc_num))
        return float(np.max(np.log(doc_num / np.asarray(dfs, dtype=float))))

    def _nqc(self, group: pd.DataFrame, depth: int = 100) -> float:
        ranked = group.sort_values("score", ascending=False).head(depth)
        variance = float(np.var(ranked["score"].to_numpy(dtype=float)))
        return variance * self._max_idf(ranked["query"].iloc[0])

    def _a_pair_ratio(self, group: pd.DataFrame, depth: int = 50) -> float:
        if self.dense_index is None:
            raise ValueError("A-Pair-Ratio requires a dense index with vec_loader().")

        from sklearn.metrics.pairwise import cosine_similarity

        depth = min(depth, len(group))
        ranked = group[group["rank"] < depth].sort_values("rank").copy()
        if ranked.empty:
            return float("nan")

        doc_vecs = self.dense_index.vec_loader()(ranked)["doc_vec"].values
        embeddings = np.vstack(doc_vecs)
        similarity = cosine_similarity(embeddings)

        scores = ranked["score"].to_numpy(dtype=float)[:, None]
        weighted = similarity @ (scores @ scores.T)

        top_n = max(1, math.ceil(0.1 * depth))
        tail_start = int(0.2 * depth)
        top_mean = float(np.mean(weighted[:top_n, :top_n]))
        tail_mean = float(np.mean(weighted[tail_start:, tail_start:]))

        if tail_mean == 0:
            return float("nan")
        return top_mean / tail_mean

    def _dense_qpp(self, group: pd.DataFrame, k: int) -> float:
        if self.dense_index is None or self.query_encoder is None:
            raise ValueError(
                "Dense QPP requires both a dense index and a query encoder."
            )

        ranked = group[group["rank"] < k].sort_values("rank").copy()
        if ranked.empty:
            return float("nan")

        query_row = ranked[["qid", "query"]].iloc[:1]
        query_vec = self.query_encoder(query_row)["query_vec"].values[0]
        doc_vecs = self.dense_index.vec_loader()(ranked)["doc_vec"].values

        embeddings = np.vstack([query_vec, *doc_vecs])
        edge = np.max(embeddings, axis=0) - np.min(embeddings, axis=0)

        # Preserve the historical spatial-QPP definition while avoiding log(0).
        edge = np.maximum(edge, np.finfo(float).tiny)
        return float(-np.sum(np.log(edge)))

    def compute(self, retrieval: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        """Return one row per retrieval query with the three CIKM QPP signals.

        Parameters
        ----------
        retrieval:
            Retrieval results containing at least ``qid``, ``query``, ``rank``
            and ``score``. Dense signals additionally require document vectors
            to be recoverable through ``dense_index.vec_loader()``.
        k:
            Number of retrieved documents used by the dense spatial QPP signal;
            the paper uses top-3 retrieval at each agentic iteration.
        """

        required = {"qid", "query", "rank", "score"}
        missing = required.difference(retrieval.columns)
        if missing:
            raise ValueError(f"retrieval dataframe is missing columns: {sorted(missing)}")

        frame = retrieval.copy()
        frame["qid"] = frame["qid"].astype(str)
        grouped = frame.groupby("qid", sort=False)

        base = frame[["qid", "query"]].drop_duplicates(subset=["qid"]).copy()

        if self.sparse_index is None:
            raise ValueError("The CIKM QPP feature set requires a sparse index for NQC.")
        if self.dense_index is None or self.query_encoder is None:
            raise ValueError(
                "The CIKM QPP feature set requires a dense index and query encoder."
            )

        nqc = grouped.apply(self._nqc).rename("nqc_score")
        a_pair = grouped.apply(self._a_pair_ratio).rename("a_pair_ratio_score")
        dense = grouped.apply(lambda g: self._dense_qpp(g, k)).rename("dense_qpp_score")

        return (
            base.merge(nqc, left_on="qid", right_index=True, how="left")
            .merge(a_pair, left_on="qid", right_index=True, how="left")
            .merge(dense, left_on="qid", right_index=True, how="left")
        )
