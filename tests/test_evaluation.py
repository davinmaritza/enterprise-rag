"""Unit tests for RAG Triad Quality Evaluation Engine."""

import unittest

from src.rag.evaluation.evaluator import RAGEvaluator


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.evaluator = RAGEvaluator(pass_threshold=0.70)
        self.sample_chunks = [
            {
                "source": "darefx_docs.md",
                "header_context": "Risk Management",
                "content": "VM 126 runs on 192.168.8.25 using Windows 10 LTSC with MT5 PID 15696. Daily drawdown threshold is 4%.",
            },
            {
                "source": "bytenodes_network.md",
                "header_context": "Network Topology",
                "content": "Node Kazuha and Xiao handle customer server containers over internal Proxmox bridge vmbr0.",
            },
        ]

    def test_context_relevance(self):
        query_relevant = "What is the IP address of VM 126 running DareFX?"
        score_high = self.evaluator.evaluate_context_relevance(query_relevant, self.sample_chunks)
        self.assertGreaterEqual(score_high, 0.40)

        query_irrelevant = "How to cook fried rice with garlic and eggs?"
        score_low = self.evaluator.evaluate_context_relevance(query_irrelevant, self.sample_chunks)
        self.assertLess(score_low, 0.30)

    def test_groundedness_vs_hallucination(self):
        # Grounded answer: facts directly exist in sample chunks
        grounded_answer = "VM 126 is located at 192.168.8.25 and runs Windows 10 LTSC [Source: darefx_docs.md]."
        grounded_score = self.evaluator.evaluate_groundedness(grounded_answer, self.sample_chunks)
        self.assertGreaterEqual(grounded_score, 0.70)

        # Hallucinated answer: claims non-existent facts
        hallucinated_answer = "VM 126 runs Fedora Linux on AWS us-east-1 with 128GB of quantum RAM."
        hallucinated_score = self.evaluator.evaluate_groundedness(hallucinated_answer, self.sample_chunks)
        self.assertLess(hallucinated_score, 0.50)

    def test_answer_relevance(self):
        query = "What is the IP address of VM 126?"
        good_answer = "The IP address of VM 126 is 192.168.8.25."
        good_score = self.evaluator.evaluate_answer_relevance(query, good_answer)
        self.assertGreaterEqual(good_score, 0.60)

    def test_evaluate_sample_pass_fail(self):
        query = "What is the IP address of VM 126 running DareFX?"
        answer = "VM 126 is running on IP 192.168.8.25 with MT5 PID 15696 [Source: darefx_docs.md]."
        res = self.evaluator.evaluate_sample(
            query=query,
            context_chunks=self.sample_chunks,
            answer=answer,
            citations=["darefx_docs.md"],
        )
        self.assertTrue(res.passed)
        self.assertGreaterEqual(res.overall_score, 0.70)


if __name__ == "__main__":
    unittest.main()
