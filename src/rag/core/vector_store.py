"""Qdrant Vector Database Store module for Enterprise RAG.

Handles collection lifecycle, HNSW vector indexing, point payload management,
and filtered vector similarity queries.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid5, NAMESPACE_DNS

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.rag.chunking.splitter import TextChunk

logger = logging.getLogger(__name__)


class QdrantStore:
    """Manages collections and vector operations within Qdrant."""

    DEFAULT_COLLECTION = "enterprise_knowledge"

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str | None = None,
        vector_size: int = 384,
    ) -> None:
        self.url = url
        self.collection_name = collection_name or self.DEFAULT_COLLECTION
        self.vector_size = vector_size
        self.client = QdrantClient(url=self.url)
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """Create collection if it does not already exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            logger.info("Creating Qdrant collection '%s' (vector size %d, COSINE distance)...",
                        self.collection_name, self.vector_size)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            # Create payload indices for fast metadata filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source",
                field_schema="keyword",
            )
        else:
            logger.info("Qdrant collection '%s' already exists.", self.collection_name)

    def upsert_chunks(
        self,
        chunks: list[TextChunk],
        vectors: list[list[float]],
    ) -> int:
        """Batch upsert text chunks with corresponding dense vectors."""
        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors.")

        points: list[PointStruct] = []
        for chunk, vec in zip(chunks, vectors):
            # Deterministic UUID from chunk_id for idempotency
            point_id = str(uuid5(NAMESPACE_DNS, chunk.chunk_id))
            payload = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "source": chunk.source,
                "doc_title": chunk.doc_title,
                "chunk_index": chunk.chunk_index,
                "header_context": chunk.header_context,
                **chunk.metadata,
            }
            points.append(PointStruct(id=point_id, vector=vec, payload=payload))

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            logger.info("Successfully upserted %d points to collection '%s'",
                        len(points), self.collection_name)

        return len(points)

    def search_dense(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_source: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search nearest neighbors using dense cosine similarity."""
        query_filter = None
        if filter_source:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=filter_source),
                    )
                ]
            )

        # In modern qdrant-client >= 1.10, query_points or search can be used
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            ).points
        except Exception:
            # Fallback for search()
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

        output: list[dict[str, Any]] = []
        for hit in results:
            payload = hit.payload or {}
            output.append({
                "chunk_id": payload.get("chunk_id", str(hit.id)),
                "content": payload.get("content", ""),
                "source": payload.get("source", ""),
                "doc_title": payload.get("doc_title", ""),
                "header_context": payload.get("header_context", ""),
                "score": hit.score,
            })
        return output

    def get_all_chunks(self, limit: int = 2000) -> list[dict[str, Any]]:
        """Fetch all chunks and payloads for sparse index initialization."""
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {
                "chunk_id": r.payload.get("chunk_id", str(r.id)),
                "content": r.payload.get("content", ""),
                "source": r.payload.get("source", ""),
                "doc_title": r.payload.get("doc_title", ""),
                "header_context": r.payload.get("header_context", ""),
            }
            for r in records
            if r.payload
        ]

    def count(self) -> int:
        """Return total indexed vectors."""
        res = self.client.count(collection_name=self.collection_name)
        return res.count
