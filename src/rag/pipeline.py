"""Enterprise Instrumented RAG Pipeline.

Coordinates the full RAG lifecycle with end-to-end distributed tracing,
recording timing waterfalls, intermediate chunk state, and token economics.
"""

from __future__ import annotations

import logging
from typing import Any

from src.rag.core.embeddings import FastEmbedder
from src.rag.core.vector_store import QdrantStore
from src.rag.observability.tracer import Tracer, default_tracer
from src.rag.search.bm25_searcher import BM25Searcher
from src.rag.search.hybrid_searcher import HybridSearcher
from src.rag.search.reranker import Reranker
from src.rag.synthesizer.generator import RAGSynthesizer

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Instrumented end-to-end RAG orchestrator."""

    def __init__(
        self,
        vector_store: QdrantStore | None = None,
        embedder: FastEmbedder | None = None,
        bm25_searcher: BM25Searcher | None = None,
        reranker: Reranker | None = None,
        synthesizer: RAGSynthesizer | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.vector_store = vector_store or QdrantStore()
        self.embedder = embedder or FastEmbedder()
        self.bm25_searcher = bm25_searcher or BM25Searcher()
        self.hybrid = HybridSearcher(
            vector_store=self.vector_store,
            embedder=self.embedder,
            bm25_searcher=self.bm25_searcher,
        )
        self.reranker = reranker or Reranker()
        self.synthesizer = synthesizer or RAGSynthesizer()
        self.tracer = tracer or default_tracer

    def query(
        self,
        question: str,
        top_k: int = 3,
        filter_source: str | None = None,
    ) -> dict[str, Any]:
        """Execute the full RAG pipeline under an active OpenTelemetry-compatible trace."""
        metadata = {
            "query": question,
            "top_k": top_k,
            "filter_source": filter_source,
            "model": self.synthesizer.model,
        }

        with self.tracer.start_trace(name="rag_pipeline_execution", metadata=metadata) as trace:
            # Stage 1: Dense Retrieval
            with self.tracer.start_span("dense_vector_search", attributes={"query": question, "top_k": top_k * 2}) as span:
                query_vector = self.embedder.embed_query(question)
                dense_hits = self.vector_store.search_dense(
                    query_vector=query_vector,
                    top_k=top_k * 2,
                    filter_source=filter_source,
                )
                span.set_attribute("hits_returned", len(dense_hits))

            # Stage 2: Sparse BM25 Keyword Search
            with self.tracer.start_span("sparse_bm25_search", attributes={"query": question, "top_k": top_k * 2}) as span:
                sparse_hits = self.bm25_searcher.search_sparse(
                    query=question,
                    top_k=top_k * 2,
                )
                span.set_attribute("hits_returned", len(sparse_hits))

            # Stage 3: Reciprocal Rank Fusion (RRF)
            with self.tracer.start_span("rrf_fusion", attributes={"dense_count": len(dense_hits), "sparse_count": len(sparse_hits)}) as span:
                scores: dict[str, float] = {}
                chunk_map: dict[str, dict[str, Any]] = {}
                rrf_k = self.hybrid.rrf_k

                for rank, hit in enumerate(dense_hits, start=1):
                    cid = hit["chunk_id"]
                    chunk_map[cid] = hit
                    scores[cid] = scores.get(cid, 0.0) + 0.6 * (1.0 / (rrf_k + rank))

                for rank, hit in enumerate(sparse_hits, start=1):
                    cid = hit["chunk_id"]
                    if cid not in chunk_map:
                        chunk_map[cid] = hit
                    scores[cid] = scores.get(cid, 0.0) + 0.4 * (1.0 / (rrf_k + rank))

                ranked_cids = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[: top_k * 2]
                raw_hits = []
                for cid in ranked_cids:
                    item = dict(chunk_map[cid])
                    item["rrf_score"] = scores[cid]
                    raw_hits.append(item)
                span.set_attribute("fused_candidates", len(raw_hits))

            # Stage 4: Cross-Relevance Re-Ranking
            with self.tracer.start_span("rerank_candidates", attributes={"candidates_count": len(raw_hits)}) as span:
                ranked_hits = self.reranker.rerank(
                    query=question,
                    candidates=raw_hits,
                    top_k=top_k,
                )
                span.set_attribute("final_chunks_selected", len(ranked_hits))

            # Stage 5: Grounded LLM Answer Generation
            with self.tracer.start_span("llm_generation", attributes={"model": self.synthesizer.model}) as span:
                result = self.synthesizer.generate(query=question, context_chunks=ranked_hits)
                span.set_attribute("citations", result.get("citations", []))

        return {
            "query": question,
            "answer": result["answer"],
            "model": result["model"],
            "citations": result.get("citations", []),
            "context_chunks": ranked_hits,
            "usage": result.get("usage", {}),
            "trace_id": trace.trace_id,
            "duration_ms": trace.duration_ms,
            "cost_usd": trace.total_cost_usd,
        }
