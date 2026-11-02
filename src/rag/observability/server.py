"""Enterprise LLM Observability Web Dashboard and REST Telemetry API.

Provides real-time waterfall trace inspection, token economics,
and interactive RAG Triad evaluation dashboard.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.rag.evaluation.evaluator import RAGEvaluator
from src.rag.observability.tracer import TraceStore, default_tracer
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise RAG Observability & Tracing Hub",
    description="Real-time telemetry, waterfall trace visualization, and RAG Triad evaluation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()
store = default_tracer.store
evaluator = RAGEvaluator()

latest_eval_cache: dict[str, Any] = {}


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@app.get("/api/stats")
def get_stats() -> dict[str, Any]:
    """Return aggregated telemetry and token economics."""
    metrics = store.get_summary_metrics()
    metrics["latest_eval"] = latest_eval_cache.get("pass_rate_pct", "Not Run")
    metrics["eval_avg_score"] = latest_eval_cache.get("avg_overall_score", 0.0)
    return metrics


@app.get("/api/traces")
def list_traces(
    limit: int = Query(default=30, ge=1, le=100),
    status: str | None = None,
) -> list[dict[str, Any]]:
    """List recent execution traces."""
    return store.list_traces(limit=limit, status=status)


@app.get("/api/traces/{trace_id}")
def get_trace(trace_id: str) -> dict[str, Any]:
    """Retrieve full trace details including span waterfall hierarchy."""
    trace_data = store.get_trace(trace_id)
    if not trace_data:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found.")
    return trace_data


@app.post("/api/query")
def execute_query(req: QueryRequest) -> dict[str, Any]:
    """Execute live RAG query under active tracing."""
    result = pipeline.query(question=req.query, top_k=req.top_k)
    return result


@app.post("/api/evaluation/run")
def run_evaluation() -> dict[str, Any]:
    """Trigger automated RAG Triad benchmark suite."""
    global latest_eval_cache
    result = evaluator.run_benchmark(pipeline)
    latest_eval_cache = result
    return result


@app.get("/api/evaluation/latest")
def get_latest_evaluation() -> dict[str, Any]:
    """Get the cached scorecard from the latest benchmark run."""
    return latest_eval_cache or {"status": "No evaluations executed yet."}


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise LLM Observability & Tracing Cockpit</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bn-bg-base: #141211;
            --bn-bg-surface: #1c1917;
            --bn-bg-card: #25211e;
            --bn-bg-card-hover: #302b27;
            --bn-mint: #EBF5EE;
            --bn-khaki: #BFA89E;
            --bn-taupe: #8B786D;
            --bn-success: #52b788;
            --bn-danger: #e63946;
            --bn-border-subtle: rgba(191, 168, 158, 0.15);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bn-bg-base);
            color: var(--bn-mint);
            font-family: 'Outfit', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            background-color: var(--bn-bg-surface);
            border-bottom: 1px solid var(--bn-border-subtle);
            padding: 16px 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .brand-title {
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .badge-live {
            background-color: rgba(82, 183, 136, 0.15);
            color: var(--bn-success);
            border: 1px solid rgba(82, 183, 136, 0.3);
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        .actions-group {
            display: flex;
            gap: 12px;
        }

        button.btn-primary {
            background-color: var(--bn-khaki);
            color: var(--bn-bg-base);
            border: none;
            padding: 8px 18px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        button.btn-primary:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }

        button.btn-secondary {
            background-color: var(--bn-bg-card);
            color: var(--bn-mint);
            border: 1px solid var(--bn-border-subtle);
            padding: 8px 18px;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        button.btn-secondary:hover {
            background-color: var(--bn-bg-card-hover);
        }

        main {
            padding: 24px 32px;
            display: flex;
            flex-direction: column;
            gap: 24px;
            flex: 1;
        }

        /* Bento Grid Stats */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
        }

        .stat-card {
            background-color: var(--bn-bg-surface);
            border: 1px solid var(--bn-border-subtle);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .stat-label {
            font-size: 13px;
            color: var(--bn-taupe);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stat-value {
            font-size: 28px;
            font-weight: 700;
            color: var(--bn-mint);
            font-family: 'JetBrains Mono', monospace;
        }

        .stat-meta {
            font-size: 12px;
            color: var(--bn-khaki);
        }

        /* Split Workspace: Traces List & Waterfall Inspector */
        .split-workspace {
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 20px;
            flex: 1;
            min-height: 520px;
        }

        .panel-container {
            background-color: var(--bn-bg-surface);
            border: 1px solid var(--bn-border-subtle);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .panel-header {
            padding: 14px 20px;
            border-bottom: 1px solid var(--bn-border-subtle);
            font-weight: 600;
            font-size: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .trace-list {
            overflow-y: auto;
            flex: 1;
            max-height: 600px;
        }

        .trace-item {
            padding: 14px 20px;
            border-bottom: 1px solid rgba(191, 168, 158, 0.08);
            cursor: pointer;
            transition: background 0.15s ease;
        }

        .trace-item:hover, .trace-item.active {
            background-color: var(--bn-bg-card-hover);
        }

        .trace-item-title {
            font-weight: 600;
            font-size: 14px;
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
        }

        .trace-item-meta {
            font-size: 12px;
            color: var(--bn-taupe);
            font-family: 'JetBrains Mono', monospace;
            display: flex;
            gap: 12px;
        }

        .status-pill {
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }

        .status-pill.ok {
            background: rgba(82, 183, 136, 0.15);
            color: var(--bn-success);
        }

        .status-pill.error {
            background: rgba(230, 57, 70, 0.15);
            color: var(--bn-danger);
        }

        /* Waterfall View */
        .waterfall-container {
            padding: 20px;
            overflow-y: auto;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .span-row {
            background-color: var(--bn-bg-card);
            border: 1px solid var(--bn-border-subtle);
            border-radius: 8px;
            padding: 12px 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .span-header {
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            font-weight: 600;
        }

        .span-bar-wrapper {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            height: 10px;
            position: relative;
            overflow: hidden;
        }

        .span-bar {
            height: 100%;
            border-radius: 4px;
            background: linear-gradient(90deg, var(--bn-khaki), var(--bn-success));
        }

        .span-details {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--bn-taupe);
            background: #12100f;
            padding: 8px 12px;
            border-radius: 6px;
            overflow-x: auto;
        }

        /* Interactive Live Query Bar */
        .live-query-bar {
            background-color: var(--bn-bg-surface);
            border: 1px solid var(--bn-border-subtle);
            border-radius: 12px;
            padding: 16px 20px;
            display: flex;
            gap: 12px;
        }

        .live-query-input {
            flex: 1;
            background: var(--bn-bg-base);
            border: 1px solid var(--bn-border-subtle);
            border-radius: 8px;
            padding: 10px 16px;
            color: var(--bn-mint);
            font-size: 14px;
            font-family: inherit;
        }

        .live-query-input:focus {
            outline: none;
            border-color: var(--bn-khaki);
        }
    </style>
</head>
<body>

    <header>
        <div class="brand-title">
            <span>🛡️ Enterprise LLM Observability</span>
            <span class="badge-live">LIVE TELEMETRY</span>
        </div>
        <div class="actions-group">
            <button class="btn-secondary" onclick="runEvaluation()">⚡ Run RAG Triad Benchmark</button>
            <button class="btn-primary" onclick="loadDashboard()">🔄 Refresh Telemetry</button>
        </div>
    </header>

    <main>
        <!-- Bento Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Traces Recorded</div>
                <div class="stat-value" id="stat-total-traces">-</div>
                <div class="stat-meta" id="stat-error-rate">Error Rate: 0.0%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average Latency</div>
                <div class="stat-value" id="stat-avg-latency">- ms</div>
                <div class="stat-meta">Full pipeline roundtrip</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Tokens & Est. Cost</div>
                <div class="stat-value" id="stat-tokens">-</div>
                <div class="stat-meta" id="stat-cost">Cost: $0.000000</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">RAG Triad Quality Score</div>
                <div class="stat-value" id="stat-eval-score">-</div>
                <div class="stat-meta" id="stat-eval-pass">Benchmark: Not Run</div>
            </div>
        </div>

        <!-- Live Query Interactive Bar -->
        <div class="live-query-bar">
            <input type="text" id="query-input" class="live-query-input" placeholder="Ask knowledge base (e.g. 'What is the IP address of VM 126 running DareFX?')...">
            <button class="btn-primary" onclick="submitLiveQuery()">Execute & Trace</button>
        </div>

        <!-- Split Workspace -->
        <div class="split-workspace">
            <!-- Left: Traces List -->
            <div class="panel-container">
                <div class="panel-header">
                    <span>Recent Execution Traces</span>
                    <span id="trace-count-badge" style="font-size:12px; color:var(--bn-taupe)">0 traces</span>
                </div>
                <div class="trace-list" id="trace-list-container">
                    <div style="padding: 20px; color: var(--bn-taupe); text-align: center;">Loading traces...</div>
                </div>
            </div>

            <!-- Right: Waterfall Inspector -->
            <div class="panel-container">
                <div class="panel-header">
                    <span id="selected-trace-title">Span Waterfall Timeline & Latency Breakdown</span>
                    <span id="selected-trace-id" style="font-family:'JetBrains Mono'; font-size:12px; color:var(--bn-khaki)">Select a trace</span>
                </div>
                <div class="waterfall-container" id="waterfall-container">
                    <div style="padding: 40px; color: var(--bn-taupe); text-align: center;">
                        Select an execution trace from the left panel to inspect its distributed span waterfall.
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        let currentTraceId = null;

        async function loadDashboard() {
            try {
                // 1. Fetch Stats
                const statsRes = await fetch('/api/stats');
                const stats = await statsRes.json();
                document.getElementById('stat-total-traces').innerText = stats.total_traces || 0;
                document.getElementById('stat-error-rate').innerText = `Error Rate: ${stats.error_rate_pct || 0}%`;
                document.getElementById('stat-avg-latency').innerText = `${stats.avg_duration_ms || 0} ms`;
                document.getElementById('stat-tokens').innerText = (stats.total_tokens_used || 0).toLocaleString();
                document.getElementById('stat-cost').innerText = `Cost: $${(stats.total_cost_usd || 0).toFixed(6)}`;

                if (stats.latest_eval !== "Not Run") {
                    document.getElementById('stat-eval-score').innerText = `${(stats.eval_avg_score * 100).toFixed(1)}%`;
                    document.getElementById('stat-eval-pass').innerText = `Pass Rate: ${stats.latest_eval}%`;
                }

                // 2. Fetch Traces
                const tracesRes = await fetch('/api/traces?limit=30');
                const traces = await tracesRes.json();
                renderTraceList(traces);

                if (traces.length > 0 && !currentTraceId) {
                    selectTrace(traces[0].trace_id);
                }
            } catch (err) {
                console.error("Failed to load telemetry dashboard", err);
            }
        }

        function renderTraceList(traces) {
            const container = document.getElementById('trace-list-container');
            document.getElementById('trace-count-badge').innerText = `${traces.length} traces`;

            if (!traces || traces.length === 0) {
                container.innerHTML = '<div style="padding: 20px; color: var(--bn-taupe); text-align: center;">No traces recorded yet. Execute a query above!</div>';
                return;
            }

            container.innerHTML = traces.map(t => {
                const isError = t.status === 'ERROR';
                return `
                    <div class="trace-item ${t.trace_id === currentTraceId ? 'active' : ''}" onclick="selectTrace('${t.trace_id}')">
                        <div class="trace-item-title">
                            <span>${t.name}</span>
                            <span class="status-pill ${isError ? 'error' : 'ok'}">${t.status}</span>
                        </div>
                        <div class="trace-item-meta">
                            <span>⏱️ ${t.duration_ms} ms</span>
                            <span>🔢 ${t.total_tokens} tokens</span>
                            <span>💵 $${t.total_cost_usd.toFixed(4)}</span>
                        </div>
                    </div>
                `;
            }).join('');
        }

        async function selectTrace(traceId) {
            currentTraceId = traceId;
            document.getElementById('selected-trace-id').innerText = `Trace: ${traceId}`;

            // Highlight active in list
            document.querySelectorAll('.trace-item').forEach(el => el.classList.remove('active'));

            try {
                const res = await fetch(`/api/traces/${traceId}`);
                const trace = await res.json();
                renderWaterfall(trace);
            } catch (err) {
                console.error("Failed to load trace", err);
            }
        }

        function renderWaterfall(trace) {
            const container = document.getElementById('waterfall-container');
            if (!trace.spans || trace.spans.length === 0) {
                container.innerHTML = '<div style="padding: 30px; color: var(--bn-taupe);">No spans found for this trace.</div>';
                return;
            }

            const totalDuration = Math.max(trace.duration_ms, 1.0);

            container.innerHTML = trace.spans.map(span => {
                const offsetMs = Math.max(0, (span.start_time - trace.start_time) * 1000.0);
                const offsetPct = Math.min(100, Math.max(0, (offsetMs / totalDuration) * 100.0));
                const widthPct = Math.min(100 - offsetPct, Math.max(3, (span.duration_ms / totalDuration) * 100.0));

                const metricsStr = span.metrics && Object.keys(span.metrics).length > 0 
                    ? ` | Metrics: ${JSON.stringify(span.metrics)}` 
                    : '';

                return `
                    <div class="span-row">
                        <div class="span-header">
                            <span>${span.name}</span>
                            <span style="font-family:'JetBrains Mono'; color:var(--bn-khaki);">${span.duration_ms} ms</span>
                        </div>
                        <div class="span-bar-wrapper">
                            <div class="span-bar" style="margin-left: ${offsetPct}%; width: ${widthPct}%;"></div>
                        </div>
                        <div class="span-details">
                            Span ID: ${span.span_id} | Parent: ${span.parent_span_id || 'root'} | Status: ${span.status}
                            <br>Attributes: ${JSON.stringify(span.attributes)}${metricsStr}
                        </div>
                    </div>
                `;
            }).join('');
        }

        async function submitLiveQuery() {
            const input = document.getElementById('query-input');
            const q = input.value.trim();
            if (!q) return;

            const btn = event.target;
            btn.innerText = "Querying...";
            btn.disabled = true;

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: q, top_k: 3})
                });
                const data = await res.json();
                await loadDashboard();
                if (data.trace_id) {
                    selectTrace(data.trace_id);
                }
            } catch (err) {
                alert("Query failed: " + err);
            } finally {
                btn.innerText = "Execute & Trace";
                btn.disabled = false;
            }
        }

        async function runEvaluation() {
            const btn = event.target;
            btn.innerText = "Evaluating...";
            btn.disabled = true;

            try {
                const res = await fetch('/api/evaluation/run', {method: 'POST'});
                const evalData = await res.json();
                alert(`RAG Triad Benchmark Complete!\n\nPass Rate: ${evalData.pass_rate_pct}%\nAvg Groundedness: ${(evalData.avg_groundedness * 100).toFixed(1)}%\nAvg Context Relevance: ${(evalData.avg_context_relevance * 100).toFixed(1)}%\nBenchmark Status: ${evalData.benchmark_status}`);
                await loadDashboard();
            } catch (err) {
                alert("Evaluation failed: " + err);
            } finally {
                btn.innerText = "⚡ Run RAG Triad Benchmark";
                btn.disabled = false;
            }
        }

        // Initialize dashboard on page load
        loadDashboard();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Render the master web dashboard."""
    return DASHBOARD_HTML
