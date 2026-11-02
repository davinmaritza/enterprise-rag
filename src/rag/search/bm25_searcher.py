"""Sparse keyword searcher using BM25 Okapi algorithm.

Complements dense semantic embeddings by providing exact token and keyword
matching for codes, error numbers, and precise terminology.
"""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


class BM25Searcher:
    """In-memory BM25 Okapi sparse search engine."""

    def __init__(self) -> None:
        self.corpus: list[dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None

    def _tokenize(self, text: str) -> list[str]:
        """Simple alphanumeric tokenizer with lowercase normalization."""
        return re.findall(r"\b\w+\b", text.lower())

    def index_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Build the BM25 inverted index from a corpus of chunk dictionaries."""
        self.corpus = chunks
        tokenized_corpus = [self._tokenize(c["content"]) for c in chunks]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def search_sparse(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search top-K chunks matching query terms using BM25 scoring."""
        if not self.bm25 or not self.corpus:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        doc_scores = self.bm25.get_scores(tokens)
        scored_indices = sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[dict[str, Any]] = []
        for idx, score in scored_indices:
            if score <= 0.0:
                continue
            item = dict(self.corpus[idx])
            item["score"] = float(score)
            results.append(item)

        return results
