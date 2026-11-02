"""Unit tests for Enterprise LLM Observability & Tracing Engine."""

import shutil
import tempfile
import unittest
from pathlib import Path

from src.rag.observability.tracer import (
    Span,
    Trace,
    Tracer,
    TraceStore,
    calculate_cost,
)


class TestObservability(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_traces.db"
        self.store = TraceStore(db_path=self.db_path)
        self.tracer = Tracer(store=self.store)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_cost_calculation(self):
        # Local models cost $0.0
        self.assertEqual(calculate_cost("qwen2.5:latest", 1000, 500), 0.0)
        self.assertEqual(calculate_cost("local_fallback", 1000, 500), 0.0)

        # Commercial models have correct rates
        # gpt-4o: $0.005/1k in, $0.015/1k out -> 1000 in ($0.005) + 1000 out ($0.015) = $0.02
        cost_gpt = calculate_cost("gpt-4o", 1000, 1000)
        self.assertAlmostEqual(cost_gpt, 0.02, places=4)

    def test_trace_lifecycle_and_spans(self):
        with self.tracer.start_trace("test_pipeline", metadata={"user": "admin"}) as trace:
            self.assertEqual(trace.name, "test_pipeline")
            self.assertEqual(trace.status, "OK")

            # Child Span 1: retrieval
            with self.tracer.start_span("retrieval", attributes={"query": "hello"}) as span1:
                span1.set_attribute("candidates", 5)

            # Child Span 2: generation
            with self.tracer.start_span("generation") as span2:
                self.tracer.record_llm_metrics("gpt-4o-mini", prompt_tokens=200, completion_tokens=100)

        self.assertGreater(trace.duration_ms, 0.0)
        self.assertEqual(len(trace.spans), 2)
        self.assertEqual(trace.total_prompt_tokens, 200)
        self.assertEqual(trace.total_completion_tokens, 100)
        self.assertEqual(trace.total_tokens, 300)
        self.assertGreater(trace.total_cost_usd, 0.0)

        # Verify persistence in SQLite
        saved = self.store.get_trace(trace.trace_id)
        self.assertIsNotNone(saved)
        self.assertEqual(saved["name"], "test_pipeline")
        self.assertEqual(len(saved["spans"]), 2)

    def test_error_handling_in_span(self):
        try:
            with self.tracer.start_trace("failing_pipeline") as trace:
                with self.tracer.start_span("critical_step") as span:
                    raise ValueError("Connection failed")
        except ValueError:
            pass

        self.assertEqual(trace.status, "ERROR")
        saved = self.store.get_trace(trace.trace_id)
        self.assertEqual(saved["status"], "ERROR")
        self.assertEqual(saved["spans"][0]["status"], "ERROR")
        self.assertIn("Connection failed", saved["spans"][0]["error_message"])

    def test_summary_metrics(self):
        with self.tracer.start_trace("trace_1") as t1:
            with self.tracer.start_span("s1"):
                self.tracer.record_llm_metrics("gpt-4o", prompt_tokens=500, completion_tokens=200)

        metrics = self.store.get_summary_metrics()
        self.assertEqual(metrics["total_traces"], 1)
        self.assertEqual(metrics["error_count"], 0)
        self.assertEqual(metrics["total_tokens_used"], 700)
        self.assertGreater(metrics["total_cost_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
