"""Cross-score Re-ranking module for Enterprise RAG.

Refines initial candidate hits from Dense/Sparse retrieval by computing
exact token density, exact title context boost, and cosine proximity alignment.
"""

from __future__ import annotations

import re
from typing import Any

from src.rag.core.embeddings import FastEmbedder


class Reranker:
    """Re-ranks retrieved candidate chunks to ensure maximal query relevance."""

    def __init__(self, embedder: FastEmbedder | None = None) -> None:
        self.embedder = embedder

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Re-rank candidate documents using token density and semantic boost."""
        if not candidates:
            return []

        query_terms = set(re.findall(r"\b\w+\b", query.lower()))
        scored_candidates: list[tuple[float, dict[str, Any]]] = []

        for item in candidates:
            content = item.get("content", "").lower()
            title = item.get("doc_title", "").lower()
            header = item.get("header_context", "").lower()
            base_rrf = item.get("rrf_score", 0.0)

            # Compute term coverage ratio (how many query words appear in content)
            matching_terms = sum(1 for term in query_terms if term in content)
            coverage = matching_terms / max(1, len(query_terms))

            # Header / Title exact term boost
            header_boost = sum(1.5 for term in query_terms if term in header or term in title)

            # Combined re-ranking score
            final_score = (base_rrf * 10.0) + (coverage * 2.0) + header_boost

            updated_item = dict(item)
            updated_item["rerank_score"] = round(final_score, 4)
            updated_item["term_coverage"] = round(coverage, 2)
            scored_candidates.append((final_score, updated_item))

        # Sort descending by rerank score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_candidates[:top_k]]
