"""Unit tests for Enterprise Hybrid RAG modules using standard unittest."""

import unittest

from src.rag.chunking.splitter import MarkdownTextSplitter, TextChunk
from src.rag.search.bm25_searcher import BM25Searcher
from src.rag.search.reranker import Reranker
from src.rag.synthesizer.generator import RAGSynthesizer


class TestEnterpriseRAG(unittest.TestCase):
    def test_markdown_splitter(self) -> None:
        text = (
            "# Title\n\n"
            "Introduction paragraph.\n\n"
            "## Section 1\n\n"
            "This is details about Section 1 containing important information."
        )
        splitter = MarkdownTextSplitter(chunk_size=60, chunk_overlap=10)
        chunks = splitter.split_text(text, source="test.md", title="Test Doc")

        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(c, TextChunk) for c in chunks))
        self.assertEqual(chunks[0].source, "test.md")

    def test_bm25_searcher(self) -> None:
        chunks = [
            {"chunk_id": "c1", "content": "The quick brown fox jumps over the lazy dog", "source": "f1"},
            {"chunk_id": "c2", "content": "MetaTrader 5 algorithmic trading and forex bot", "source": "f2"},
            {"chunk_id": "c3", "content": "Bytenodes Proxmox virtualization and LXC clusters", "source": "f3"},
        ]
        searcher = BM25Searcher()
        searcher.index_chunks(chunks)

        res = searcher.search_sparse("MetaTrader bot", top_k=2)
        self.assertGreater(len(res), 0)
        self.assertEqual(res[0]["chunk_id"], "c2")

    def test_reranker(self) -> None:
        candidates = [
            {"chunk_id": "1", "content": "Generic unrelated text", "doc_title": "General", "header_context": "", "rrf_score": 0.01},
            {"chunk_id": "2", "content": "Exact match for Proxmox cluster nodes and host IP", "doc_title": "Proxmox", "header_context": "Proxmox Cluster", "rrf_score": 0.05},
        ]
        reranker = Reranker()
        ranked = reranker.rerank("Proxmox cluster host IP", candidates, top_k=2)

        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["chunk_id"], "2")
        self.assertGreater(ranked[0]["rerank_score"], ranked[1]["rerank_score"])

    def test_synthesizer_context_builder(self) -> None:
        synthesizer = RAGSynthesizer()
        chunks = [
            {"chunk_id": "abc1", "source": "doc1.md", "header_context": "Header A", "content": "Content snippet 1"},
            {"chunk_id": "abc2", "source": "doc2.md", "header_context": "Header B", "content": "Content snippet 2"},
        ]
        block = synthesizer.build_context_block(chunks)

        self.assertIn("[Document #1] Source: doc1.md (Chunk ID: abc1) > Header A", block)
        self.assertIn("Content snippet 1", block)
        self.assertIn("[Document #2] Source: doc2.md (Chunk ID: abc2) > Header B", block)


if __name__ == "__main__":
    unittest.main()
