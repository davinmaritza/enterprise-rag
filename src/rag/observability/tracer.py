"""Enterprise LLM Observability, Distributed Tracing and Metrics Engine.

Provides OpenTelemetry-compatible Span and Trace context management,
token usage accounting, cost modeling, and SQLite WAL persistence.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Active trace and span context variables for thread/async safety
_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar("current_trace", default=None)
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar("current_span", default=None)

# Pricing matrix per 1,000 tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "qwen2.5": {"input": 0.0, "output": 0.0},
    "qwen2.5-coder:3b": {"input": 0.0, "output": 0.0},
    "local_fallback": {"input": 0.0, "output": 0.0},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "deepseek-chat": {"input": 0.00014, "output": 0.00028},
}


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate estimated API cost in USD based on model pricing matrix."""
    normalized_name = model_name.lower().split("/")[-1].split(":")[0]
    pricing = MODEL_PRICING.get(normalized_name) or MODEL_PRICING.get(model_name.lower(), {"input": 0.0, "output": 0.0})
    input_cost = (prompt_tokens / 1000.0) * pricing["input"]
    output_cost = (completion_tokens / 1000.0) * pricing["output"]
    return round(input_cost + output_cost, 6)


@dataclass
class Span:
    """Represents an atomic execution span inside a distributed trace."""

    span_id: str
    trace_id: str
    name: str
    parent_span_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float = 0.0
    status: str = "OK"  # "OK" | "ERROR"
    error_message: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def set_attribute(self, key: str, value: Any) -> None:
        """Add metadata attribute to span."""
        self.attributes[key] = value

    def set_error(self, err: Exception | str) -> None:
        """Record error status on span."""
        self.status = "ERROR"
        self.error_message = str(err)

    def finish(self) -> None:
        """Mark span as finished and calculate duration."""
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000.0, 3)

    def to_dict(self) -> dict[str, Any]:
        """Convert span to serializable dictionary."""
        return asdict(self)


@dataclass
class Trace:
    """Represents a complete lifecycle trace containing multiple child spans."""

    trace_id: str
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float = 0.0
    status: str = "OK"
    spans: list[Span] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_span(self, span: Span) -> None:
        """Add finished span and accumulate token metrics."""
        self.spans.append(span)
        if span.status == "ERROR":
            self.status = "ERROR"

        # Accumulate metrics if present
        p_tokens = span.metrics.get("prompt_tokens", 0)
        c_tokens = span.metrics.get("completion_tokens", 0)
        cost = span.metrics.get("cost_usd", 0.0)

        self.total_prompt_tokens += p_tokens
        self.total_completion_tokens += c_tokens
        self.total_tokens += p_tokens + c_tokens
        self.total_cost_usd = round(self.total_cost_usd + cost, 6)

    def finish(self) -> None:
        """Mark trace as finished."""
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000.0, 3)

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to serializable dictionary."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "metadata": self.metadata,
            "spans": [s.to_dict() for s in self.spans],
        }


class TraceStore:
    """High-performance SQLite WAL database for trace persistence."""

    def __init__(self, db_path: str | Path = "data/traces.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    duration_ms REAL,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost_usd REAL DEFAULT 0.0,
                    data_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status)")

    def save_trace(self, trace: Trace) -> None:
        """Persist a complete trace object."""
        with self._get_connection() as conn:
            data = trace.to_dict()
            conn.execute(
                """
                INSERT OR REPLACE INTO traces (
                    trace_id, name, status, start_time, end_time, duration_ms,
                    prompt_tokens, completion_tokens, total_tokens, total_cost_usd, data_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.trace_id,
                    trace.name,
                    trace.status,
                    trace.start_time,
                    trace.end_time,
                    trace.duration_ms,
                    trace.total_prompt_tokens,
                    trace.total_completion_tokens,
                    trace.total_tokens,
                    trace.total_cost_usd,
                    json.dumps(data),
                ),
            )

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Retrieve trace by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT data_json FROM traces WHERE trace_id = ?", (trace_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row["data_json"])
        return None

    def list_traces(self, limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
        """List recent traces with summary metrics."""
        query = "SELECT trace_id, name, status, duration_ms, total_tokens, total_cost_usd, created_at FROM traces"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    def get_summary_metrics(self) -> dict[str, Any]:
        """Calculate overall telemetry statistics across all runs."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_traces,
                    SUM(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END) as error_count,
                    AVG(duration_ms) as avg_duration_ms,
                    SUM(total_tokens) as total_tokens_used,
                    SUM(total_cost_usd) as total_cost_usd
                FROM traces
                """
            )
            row = dict(cursor.fetchone() or {})
            total = row.get("total_traces") or 0
            errors = row.get("error_count") or 0
            row["error_rate_pct"] = round((errors / total * 100.0) if total > 0 else 0.0, 2)
            row["avg_duration_ms"] = round(row.get("avg_duration_ms") or 0.0, 2)
            row["total_cost_usd"] = round(row.get("total_cost_usd") or 0.0, 6)
            row["total_tokens_used"] = int(row.get("total_tokens_used") or 0)
            return row


class SpanContext:
    """Context manager for an individual span."""

    def __init__(self, tracer: Tracer, span: Span) -> None:
        self.tracer = tracer
        self.span = span
        self._token: contextvars.Token | None = None

    def __enter__(self) -> Span:
        self._token = _current_span.set(self.span)
        return self.span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_val is not None:
            self.span.set_error(exc_val)
        self.span.finish()
        _current_span.reset(self._token)
        trace = _current_trace.get()
        if trace:
            trace.add_span(self.span)


class TraceContext:
    """Context manager for an entire trace."""

    def __init__(self, tracer: Tracer, trace: Trace) -> None:
        self.tracer = tracer
        self.trace = trace
        self._token: contextvars.Token | None = None

    def __enter__(self) -> Trace:
        self._token = _current_trace.set(self.trace)
        return self.trace

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_val is not None:
            self.trace.status = "ERROR"
        self.trace.finish()
        _current_trace.reset(self._token)
        self.tracer.store.save_trace(self.trace)


class Tracer:
    """Central Tracer coordinating span creation, nested hierarchy, and persistence."""

    def __init__(self, store: TraceStore | None = None) -> None:
        self.store = store or TraceStore()

    def start_trace(self, name: str, metadata: dict[str, Any] | None = None) -> TraceContext:
        """Start a new root trace."""
        trace_id = str(uuid.uuid4())[:8]
        trace = Trace(trace_id=trace_id, name=name, metadata=metadata or {})
        return TraceContext(self, trace)

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> SpanContext:
        """Start a child span within the current active trace."""
        current_t = _current_trace.get()
        trace_id = current_t.trace_id if current_t else str(uuid.uuid4())[:8]
        current_s = _current_span.get()
        parent_id = current_s.span_id if current_s else None

        span = Span(
            span_id=str(uuid.uuid4())[:8],
            trace_id=trace_id,
            name=name,
            parent_span_id=parent_id,
            attributes=attributes or {},
        )
        return SpanContext(self, span)

    def get_current_span(self) -> Span | None:
        """Return the currently executing span if any."""
        return _current_span.get()

    def record_llm_metrics(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        span: Span | None = None,
    ) -> None:
        """Record token usage and estimated cost onto target span."""
        target_span = span or _current_span.get()
        cost = calculate_cost(model, prompt_tokens, completion_tokens)
        metrics = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": cost,
        }
        if target_span:
            target_span.metrics.update(metrics)


# Global tracer instance singleton
default_tracer = Tracer()
