"""Embedding provider module for Enterprise RAG.

Utilizes FastEmbed with ONNX runtime for ultra-fast local inference (<10ms)
without requiring PyTorch or heavy C++ compilation dependencies.
"""

from __future__ import annotations

import logging
from typing import Sequence

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


class FastEmbedder:
    """Local high-throughput text embedding generator via FastEmbed & ONNX."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        logger.info("Initializing FastEmbed text embedding model: %s", self.model_name)
        self._model = TextEmbedding(model_name=self.model_name)
        # BAAI/bge-small-en-v1.5 dimension is 384
        self.vector_size = 384

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate normalized dense embedding vectors for a batch of texts."""
        if not texts:
            return []
        embeddings_generator = self._model.embed(texts)
        return [list(vec) for vec in embeddings_generator]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        results = self.embed_texts([query])
        return results[0]
