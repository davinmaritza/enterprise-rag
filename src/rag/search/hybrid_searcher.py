"""Hybrid Search Engine combining Dense Vector Search and BM25 Sparse Search.

Uses Reciprocal Rank Fusion (RRF) to blend semantic understanding with exact
lexical keyword matching into a unified relevance ranking.
"""

from __future__ import annotations

import logging
from typing import Any

from src.rag.core.embeddings import FastEmbedder
from src.rag.core.vector_store import QdrantStore
from src.rag.search.bm25_searcher import BM25Searcher

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Orchestrates multi-modal retrieval across dense vectors and sparse BM25 indices."""

    def __init__(
        self,
        vector_store: QdrantStore,
        embedder: FastEmbedder,
        bm25_searcher: BM25Searcher | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_searcher = bm25_searcher or BM25Searcher()
        self.rrf_k = rrf_k
        self._sync_bm25_corpus()

    def _sync_bm25_corpus(self) -> None:
        """Load current vector store chunks into BM25 index."""
        chunks = self.vector_store.get_all_chunks(limit=5000)
        self.bm25_searcher.index_chunks(chunks)

    def refresh_index(self) -> None:
        """Manually trigger BM25 synchronization."""
        self._sync_bm25_corpus()

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        filter_source: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute hybrid search using Reciprocal Rank Fusion (RRF)."""
        # 1. Dense retrieval via Qdrant
        query_vector = self.embedder.embed_query(query)
        dense_hits = self.vector_store.search_dense(
            query_vector=query_vector,
            top_k=top_k * 2,
            filter_source=filter_source,
        )

        # 2. Sparse retrieval via BM25
        sparse_hits = self.bm25_searcher.search_sparse(
            query=query,
            top_k=top_k * 2,
        )

        # 3. Reciprocal Rank Fusion (RRF)
        scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        # Score dense ranks
        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit["chunk_id"]
            chunk_map[cid] = hit
            scores[cid] = scores.get(cid, 0.0) + dense_weight * (1.0 / (self.rrf_k + rank))

        # Score sparse ranks
        for rank, hit in enumerate(sparse_hits, start=1):
            cid = hit["chunk_id"]
            if cid not in chunk_map:
                chunk_map[cid] = hit
            scores[cid] = scores.get(cid, 0.0) + sparse_weight * (1.0 / (self.rrf_k + rank))

        # Sort combined results by RRF score descending
        ranked_cids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]

        final_results: list[dict[str, Any]] = []
        for cid in ranked_cids:
            item = dict(chunk_map[cid])
            item["rrf_score"] = scores[cid]
            final_results.append(item)

        return final_results
