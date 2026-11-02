"""RAG Synthesizer and Context Assembler module.

Builds grounded prompts with strict citation obligations and calls
OpenAI-compatible LLM endpoints (9Router, Ollama, Gemini, DeepSeek).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class RAGSynthesizer:
    """Assembles retrieved context and coordinates LLM generation with exact citations."""

    SYSTEM_PROMPT = (
        "You are an enterprise AI assistant answering strictly from verified retrieved context.\n"
        "Rules:\n"
        "1. Base your answer ONLY on the provided context snippets.\n"
        "2. Do not invent or extrapolate facts beyond what is written.\n"
        "3. Cite every fact using the exact source format: [Source: <doc_name>, Chunk #<id>].\n"
        "4. If the context does not contain enough information to answer, state clearly:\n"
        "   'Informasi tersebut tidak ditemukan dalam basis pengetahuan saat ini.'\n"
        "5. Be direct, concise, and structured."
    )

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "ollama")
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:latest")

    def build_context_block(self, chunks: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into a numbered, citation-ready context block."""
        parts: list[str] = []
        for idx, c in enumerate(chunks, start=1):
            source = c.get("source", "Unknown")
            cid = c.get("chunk_id", str(idx))
            header = c.get("header_context", "")
            header_str = f" > {header}" if header else ""
            content = c.get("content", "").strip()

            parts.append(
                f"--- [Document #{idx}] Source: {source} (Chunk ID: {cid}){header_str} ---\n"
                f"{content}\n"
            )
        return "\n".join(parts)

    def generate(self, query: str, context_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate grounded answer using configured LLM or fallback synthesis."""
        context_str = self.build_context_block(context_chunks)
        user_prompt = (
            f"Retrieved Context:\n"
            f"{context_str}\n\n"
            f"User Question: {query}\n\n"
            f"Please answer the question based solely on the context above, citing each document used."
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    answer = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    p_tok = usage.get("prompt_tokens") or len(user_prompt.split())
                    c_tok = usage.get("completion_tokens") or len(answer.split())
                    
                    try:
                        from src.rag.observability.tracer import default_tracer
                        default_tracer.record_llm_metrics(
                            model=self.model,
                            prompt_tokens=p_tok,
                            completion_tokens=c_tok,
                        )
                    except Exception:
                        pass

                    return {
                        "answer": answer,
                        "model": self.model,
                        "used_chunks_count": len(context_chunks),
                        "citations": [c.get("source") for c in context_chunks],
                        "usage": {
                            "prompt_tokens": p_tok,
                            "completion_tokens": c_tok,
                            "total_tokens": p_tok + c_tok,
                        },
                    }
        except Exception as err:
            logger.warning("LLM generation request failed: %s. Returning structured context answer.", err)

        # Fallback offline synthesis if LLM daemon is unavailable
        fallback_answer = (
            f"Hasil Temuan dari Basis Pengetahuan ({len(context_chunks)} dokumen relevan):\n\n"
        )
        for idx, c in enumerate(context_chunks, 1):
            fallback_answer += f"**{idx}. [{c.get('source')}] {c.get('header_context', '')}**\n{c.get('content')}\n\n"

        fb_p_tok = len(user_prompt.split())
        fb_c_tok = len(fallback_answer.split())
        try:
            from src.rag.observability.tracer import default_tracer
            default_tracer.record_llm_metrics(
                model="local_fallback",
                prompt_tokens=fb_p_tok,
                completion_tokens=fb_c_tok,
            )
        except Exception:
            pass

        return {
            "answer": fallback_answer.strip(),
            "model": "local_fallback",
            "used_chunks_count": len(context_chunks),
            "citations": [c.get("source") for c in context_chunks],
            "usage": {
                "prompt_tokens": fb_p_tok,
                "completion_tokens": fb_c_tok,
                "total_tokens": fb_p_tok + fb_c_tok,
            },
        }
