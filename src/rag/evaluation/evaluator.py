"""RAG Triad Evaluation Engine.

Implements automated scoring for the three foundational RAG quality pillars:
1. Context Relevance / Precision (Retrieval Quality)
2. Groundedness / Faithfulness (Hallucination Detection)
3. Answer Relevance (Prompt Fulfillment)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class TriadResult:
    """Evaluation result for an individual test case."""

    query: str
    answer: str
    context_relevance: float
    groundedness: float
    answer_relevance: float
    overall_score: float
    passed: bool
    used_chunks: int
    citations: list[str]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RAGEvaluator:
    """Evaluates RAG pipeline outputs against the RAG Triad quality standards."""

    def __init__(self, pass_threshold: float = 0.75) -> None:
        self.pass_threshold = pass_threshold

    STOP_WORDS = {
        "what", "when", "where", "which", "with", "this", "that", "from",
        "have", "been", "does", "will", "your", "about", "into", "the",
        "is", "are", "was", "were", "of", "to", "in", "on", "at", "for",
        "and", "or", "how", "can", "could", "should", "would", "any", "all",
    }

    def evaluate_context_relevance(self, query: str, context_chunks: list[dict[str, Any]]) -> float:
        """Measure what fraction of retrieved chunks contain query key terms and concepts."""
        if not context_chunks:
            return 0.0

        raw_terms = re.findall(r"[A-Za-z0-9_-]+", query.lower())
        query_terms = [t for t in raw_terms if len(t) >= 2 and t not in self.STOP_WORDS]
        if not query_terms:
            return 1.0

        relevant_chunks = 0
        for chunk in context_chunks:
            source = chunk.get("source", "").lower()
            content = chunk.get("content", "").lower()
            header = chunk.get("header_context", "").lower()
            text = f"{source} {header} {content}"

            matches = sum(1 for term in query_terms if term in text or (len(term) > 3 and term[:4] in text))
            term_overlap_ratio = matches / len(query_terms)
            if term_overlap_ratio >= 0.25:  # Chunk addresses core query intent
                relevant_chunks += 1

        score = relevant_chunks / len(context_chunks)
        return round(min(1.0, max(0.0, score)), 3)

    def evaluate_groundedness(self, answer: str, context_chunks: list[dict[str, Any]]) -> float:
        """Verify that assertions made in the answer are backed by retrieved context (Hallucination check)."""
        if not context_chunks or not answer.strip():
            return 0.0

        combined_context = " ".join(c.get("content", "").lower() for c in context_chunks)
        sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 15]

        if not sentences:
            return 1.0

        grounded_sentences = 0
        for sent in sentences:
            # Extract factual tokens (words with >= 4 chars or numbers)
            tokens = re.findall(r"[A-Za-z0-9_-]{4,}|\d+", sent.lower())
            if not tokens:
                grounded_sentences += 1
                continue

            matched_tokens = sum(1 for tok in tokens if tok in combined_context)
            ratio = matched_tokens / len(tokens)
            if ratio >= 0.50:  # Majority of factual terms exist in retrieved context
                grounded_sentences += 1

        # Check for citation existence
        has_citations = bool(re.search(r"\[(Source|Dokumen|Document|Chunk)", answer, re.IGNORECASE))
        base_score = grounded_sentences / len(sentences)

        if has_citations:
            base_score = min(1.0, base_score + 0.10)

        return round(min(1.0, max(0.0, base_score)), 3)

    def evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """Measure how directly the answer addresses the intent of the query."""
        if not answer.strip():
            return 0.0

        query_words = set(re.findall(r"\w{3,}", query.lower()))
        answer_words = set(re.findall(r"\w{3,}", answer.lower()))

        if not query_words:
            return 1.0

        overlap = len(query_words.intersection(answer_words))
        ratio = overlap / len(query_words)

        # Penalize refusal if context was present
        if "tidak ditemukan" in answer.lower() and len(answer_words) < 20:
            return 0.5

        # Generous relevance scaling based on key intent matching
        scaled = min(1.0, ratio * 1.5 + 0.25)
        return round(scaled, 3)

    def evaluate_sample(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        answer: str,
        citations: list[str] | None = None,
    ) -> TriadResult:
        """Perform full RAG Triad evaluation on a single query and response."""
        c_rel = self.evaluate_context_relevance(query, context_chunks)
        ground = self.evaluate_groundedness(answer, context_chunks)
        a_rel = self.evaluate_answer_relevance(query, answer)

        # Overall harmonic/weighted average
        overall = round(0.35 * c_rel + 0.40 * ground + 0.25 * a_rel, 3)
        passed = overall >= self.pass_threshold

        return TriadResult(
            query=query,
            answer=answer,
            context_relevance=c_rel,
            groundedness=ground,
            answer_relevance=a_rel,
            overall_score=overall,
            passed=passed,
            used_chunks=len(context_chunks),
            citations=citations or [],
        )

    def run_benchmark(
        self,
        pipeline: RAGPipeline,
        dataset_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Execute automated benchmark suite across golden evaluation questions."""
        default_dataset = [
            {
                "query": "What is the IP address and operating system of VM 126 running DareFX?",
                "expected_topic": "darefx_trading_engine",
            },
            {
                "query": "What is the minimum balance threshold for TriPay withdrawal payouts?",
                "expected_topic": "tripay_settlement_rules",
            },
            {
                "query": "How are automated snapshots managed on the Bytenodes S3 Backup Gateway?",
                "expected_topic": "bytenodes_infrastructure",
            },
            {
                "query": "What risk mitigation rules does DareFX Aegis-Alpha enforce for daily drawdown?",
                "expected_topic": "darefx_trading_engine",
            },
            {
                "query": "Which Bank Indonesia clearing hours apply to TriPay weekend settlement?",
                "expected_topic": "tripay_settlement_rules",
            },
        ]

        if dataset_path and Path(dataset_path).exists():
            with open(dataset_path) as f:
                dataset = json.load(f)
        else:
            dataset = default_dataset

        results: list[TriadResult] = []
        for item in dataset:
            q = item["query"]
            response = pipeline.query(question=q, top_k=3)
            triad = self.evaluate_sample(
                query=q,
                context_chunks=response["context_chunks"],
                answer=response["answer"],
                citations=response.get("citations"),
            )
            results.append(triad)

        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        avg_c_rel = round(sum(r.context_relevance for r in results) / total, 3) if total else 0.0
        avg_ground = round(sum(r.groundedness for r in results) / total, 3) if total else 0.0
        avg_a_rel = round(sum(r.answer_relevance for r in results) / total, 3) if total else 0.0
        avg_overall = round(sum(r.overall_score for r in results) / total, 3) if total else 0.0

        return {
            "total_cases": total,
            "passed_cases": passed_count,
            "pass_rate_pct": round((passed_count / total) * 100.0, 1) if total else 0.0,
            "avg_context_relevance": avg_c_rel,
            "avg_groundedness": avg_ground,
            "avg_answer_relevance": avg_a_rel,
            "avg_overall_score": avg_overall,
            "benchmark_status": "PASSED" if (passed_count / total) >= 0.80 else "FAILED",
            "results": [r.to_dict() for r in results],
        }
