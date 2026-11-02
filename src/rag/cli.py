"""Enterprise RAG Command Line Interface (CLI).

Provides commands for document ingestion, collection management,
hybrid search inspection, question answering with citations,
distributed trace inspection, RAG Triad evaluation, and telemetry dashboard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.rag.chunking.splitter import MarkdownTextSplitter
from src.rag.core.embeddings import FastEmbedder
from src.rag.core.vector_store import QdrantStore
from src.rag.evaluation.evaluator import RAGEvaluator
from src.rag.observability.tracer import default_tracer
from src.rag.pipeline import RAGPipeline
from src.rag.search.hybrid_searcher import HybridSearcher
from src.rag.search.reranker import Reranker

console = Console()


def cmd_status(args: argparse.Namespace) -> None:
    """Check Qdrant health and collection statistics."""
    store = QdrantStore(url=args.url, collection_name=args.collection)
    count = store.count()

    table = Table(title="🛡️ Enterprise Qdrant Vector DB Status")
    table.add_column("Parameter", style="cyan bold")
    table.add_column("Value", style="green")

    table.add_row("Qdrant Server URL", args.url)
    table.add_row("Web Dashboard", f"{args.url}/dashboard")
    table.add_row("Collection Name", store.collection_name)
    table.add_row("Vector Dimension", f"{store.vector_size} (BGE-Small / MiniLM)")
    table.add_row("Total Vectors Indexed", str(count))
    table.add_row("Metric Distance", "COSINE")

    console.print(table)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest documents from a directory or file into Qdrant and BM25."""
    target_path = Path(args.path)
    if not target_path.exists():
        console.print(f"[red]Error: Path '{target_path}' does not exist.[/red]")
        sys.exit(1)

    files: list[Path] = []
    if target_path.is_file():
        files = [target_path]
    else:
        for ext in ("*.md", "*.txt", "*.json"):
            files.extend(target_path.glob(ext))

    if not files:
        console.print(f"[yellow]No matching files found in '{target_path}'.[/yellow]")
        return

    console.print(f"[cyan]Found {len(files)} file(s) to process. Splitting into semantic chunks...[/cyan]")
    splitter = MarkdownTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    all_chunks = []
    for f in files:
        chunks = splitter.split_file(f)
        all_chunks.extend(chunks)

    console.print(f"[green]Generated {len(all_chunks)} text chunks across all documents.[/green]")
    console.print("[cyan]Generating dense vectors via FastEmbed (ONNX)...[/cyan]")
    embedder = FastEmbedder()
    texts = [c.content for c in all_chunks]
    vectors = embedder.embed_texts(texts)

    console.print("[cyan]Upserting vectors into Qdrant...[/cyan]")
    store = QdrantStore(url=args.url, collection_name=args.collection)
    inserted = store.upsert_chunks(all_chunks, vectors)

    console.print(Panel(
        f"✅ [bold green]Ingestion Complete![/bold green]\n\n"
        f"• Files Processed: [cyan]{len(files)}[/cyan]\n"
        f"• Chunks Created: [cyan]{len(all_chunks)}[/cyan]\n"
        f"• Points Upserted: [cyan]{inserted}[/cyan]\n"
        f"• Total Collection Size: [bold cyan]{store.count()}[/bold cyan] vectors\n"
        f"• Dashboard: [blue]{args.url}/dashboard[/blue]",
        title="Ingestion Summary",
    ))


def cmd_search(args: argparse.Namespace) -> None:
    """Execute hybrid search (Dense + BM25 + RRF + Reranker) and display results."""
    query = args.query
    console.print(f"[cyan]Running Hybrid Search for query: [/cyan]'{query}'\n")

    store = QdrantStore(url=args.url, collection_name=args.collection)
    embedder = FastEmbedder()
    hybrid = HybridSearcher(vector_store=store, embedder=embedder)
    reranker = Reranker()

    raw_hits = hybrid.search_hybrid(query=query, top_k=args.top_k * 2)
    ranked_hits = reranker.rerank(query=query, candidates=raw_hits, top_k=args.top_k)

    if not ranked_hits:
        console.print("[yellow]No relevant chunks found.[/yellow]")
        return

    table = Table(title=f"🎯 Top-{len(ranked_hits)} Retrieved Chunks (Hybrid RRF + Re-Rank)")
    table.add_column("Rank", style="bold yellow", width=6)
    table.add_column("Score", style="cyan", width=10)
    table.add_column("Source", style="green", width=22)
    table.add_column("Header / Section", style="magenta", width=20)
    table.add_column("Snippet Preview", style="white")

    for idx, hit in enumerate(ranked_hits, start=1):
        score_str = f"{hit.get('rerank_score', 0.0):.3f}"
        snippet = hit.get("content", "").replace("\n", " ")[:120] + "..."
        table.add_row(
            f"#{idx}",
            score_str,
            hit.get("source", ""),
            hit.get("header_context", "")[:20],
            snippet,
        )

    console.print(table)


def cmd_ask(args: argparse.Namespace) -> None:
    """Run full instrumented RAG pipeline with automatic distributed trace logging."""
    query = args.query
    console.print(f"[cyan]Executing Instrumented RAG Pipeline for:[/cyan] '{query}'...")

    store = QdrantStore(url=args.url, collection_name=args.collection)
    pipeline = RAGPipeline(vector_store=store)
    result = pipeline.query(question=query, top_k=args.top_k)

    console.print(Panel(
        result["answer"],
        title=f"🤖 RAG Answer (Model: {result['model']})",
        subtitle=(
            f"Trace ID: {result['trace_id']} | "
            f"Latency: {result['duration_ms']:.1f}ms | "
            f"Tokens: {result['usage'].get('total_tokens', 0)} | "
            f"Cost: ${result['cost_usd']:.6f}"
        ),
    ))


def cmd_trace_list(args: argparse.Namespace) -> None:
    """Display recent execution traces."""
    traces = default_tracer.store.list_traces(limit=args.limit, status=args.status)
    if not traces:
        console.print("[yellow]No traces recorded yet.[/yellow]")
        return

    table = Table(title="🔍 Recent RAG Pipeline Distributed Traces")
    table.add_column("Trace ID", style="bold cyan")
    table.add_column("Name", style="white")
    table.add_column("Status", style="bold")
    table.add_column("Latency (ms)", style="yellow")
    table.add_column("Total Tokens", style="magenta")
    table.add_column("Est. Cost ($)", style="green")
    table.add_column("Timestamp", style="dim")

    for t in traces:
        status_style = "green" if t["status"] == "OK" else "red bold"
        table.add_row(
            t["trace_id"],
            t["name"],
            f"[{status_style}]{t['status']}[/{status_style}]",
            f"{t['duration_ms']:.1f}",
            str(t["total_tokens"]),
            f"${t['total_cost_usd']:.6f}",
            str(t.get("created_at", ""))[:19],
        )

    console.print(table)


def cmd_trace_show(args: argparse.Namespace) -> None:
    """Inspect waterfall span breakdown for a specific trace ID."""
    trace = default_tracer.store.get_trace(args.trace_id)
    if not trace:
        console.print(f"[red]Error: Trace '{args.trace_id}' not found.[/red]")
        sys.exit(1)

    console.print(Panel(
        f"[bold]Trace Name:[/bold] {trace['name']}\n"
        f"[bold]Status:[/bold] {trace['status']}\n"
        f"[bold]Total Duration:[/bold] {trace['duration_ms']:.2f} ms\n"
        f"[bold]Total Tokens:[/bold] {trace['total_tokens']} (Prompt: {trace['total_prompt_tokens']}, Completion: {trace['total_completion_tokens']})\n"
        f"[bold]Estimated Cost:[/bold] ${trace['total_cost_usd']:.6f}\n"
        f"[bold]Metadata:[/bold] {trace.get('metadata')}",
        title=f"🔬 Trace Details: {trace['trace_id']}",
    ))

    table = Table(title="🌊 Span Execution Waterfall")
    table.add_column("Span ID", style="cyan", width=10)
    table.add_column("Parent ID", style="dim", width=10)
    table.add_column("Span Name", style="bold white", width=24)
    table.add_column("Status", width=8)
    table.add_column("Duration", style="yellow", width=12)
    table.add_column("Key Attributes & Metrics", style="white")

    for span in trace.get("spans", []):
        status_style = "green" if span["status"] == "OK" else "red bold"
        attr_summary = ", ".join(f"{k}={v}" for k, v in span.get("attributes", {}).items() if k not in ("content", "chunks"))
        if span.get("metrics"):
            attr_summary += f" | {span['metrics']}"
        table.add_row(
            span["span_id"],
            span.get("parent_span_id") or "root",
            span["name"],
            f"[{status_style}]{span['status']}[/{status_style}]",
            f"{span['duration_ms']:.2f} ms",
            attr_summary[:100],
        )

    console.print(table)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run automated RAG Triad benchmark suite."""
    console.print("[cyan]Running automated RAG Triad Quality Benchmark...[/cyan]\n")

    store = QdrantStore(url=args.url, collection_name=args.collection)
    pipeline = RAGPipeline(vector_store=store)
    evaluator = RAGEvaluator(pass_threshold=args.threshold)

    results = evaluator.run_benchmark(pipeline)

    table = Table(title="📊 RAG Triad Evaluation Scorecard")
    table.add_column("Test Query", style="cyan", width=34)
    table.add_column("Context Relevance", style="yellow", width=18)
    table.add_column("Groundedness", style="green", width=15)
    table.add_column("Answer Relevance", style="magenta", width=18)
    table.add_column("Overall", style="bold white", width=10)
    table.add_column("Result", width=8)

    for r in results["results"]:
        status_str = "[bold green]PASS[/bold green]" if r["passed"] else "[bold red]FAIL[/bold red]"
        table.add_row(
            r["query"][:32] + "...",
            f"{r['context_relevance'] * 100:.1f}%",
            f"{r['groundedness'] * 100:.1f}%",
            f"{r['answer_relevance'] * 100:.1f}%",
            f"{r['overall_score'] * 100:.1f}%",
            status_str,
        )

    console.print(table)

    summary_color = "green" if results["benchmark_status"] == "PASSED" else "red"
    console.print(Panel(
        f"• [bold]Benchmark Status:[/bold] [{summary_color}]{results['benchmark_status']}[/{summary_color}]\n"
        f"• [bold]Pass Rate:[/bold] [bold cyan]{results['pass_rate_pct']}%[/bold cyan] ({results['passed_cases']}/{results['total_cases']} cases)\n"
        f"• [bold]Average Context Relevance:[/bold] {results['avg_context_relevance'] * 100:.1f}%\n"
        f"• [bold]Average Groundedness (Anti-Hallucination):[/bold] {results['avg_groundedness'] * 100:.1f}%\n"
        f"• [bold]Average Answer Relevance:[/bold] {results['avg_answer_relevance'] * 100:.1f}%\n"
        f"• [bold]Overall Triad Quality Index:[/bold] [bold {summary_color}]{results['avg_overall_score'] * 100:.1f}%[/bold {summary_color}]",
        title="Benchmark Summary",
    ))


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the live Observability Web Dashboard."""
    import uvicorn
    console.print(f"[bold green]Starting Observability Cockpit at http://localhost:{args.port}...[/bold green]")
    uvicorn.run("src.rag.observability.server:app", host="0.0.0.0", port=args.port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise Hybrid RAG CLI & Observability Hub")
    parser.add_argument("--url", default="http://localhost:6333", help="Qdrant service URL")
    parser.add_argument("--collection", default="enterprise_knowledge", help="Collection name")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    subparsers.add_parser("status", help="Inspect Qdrant collection status")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents into Qdrant")
    p_ingest.add_argument("--path", required=True, help="Directory or file path to ingest")
    p_ingest.add_argument("--chunk-size", type=int, default=450, help="Chunk character size")
    p_ingest.add_argument("--chunk-overlap", type=int, default=60, help="Chunk overlap size")

    # search
    p_search = subparsers.add_parser("search", help="Execute hybrid search inspection")
    p_search.add_argument("query", help="Query string")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    # ask
    p_ask = subparsers.add_parser("ask", help="Ask a question with grounded citations & distributed tracing")
    p_ask.add_argument("query", help="Question prompt")
    p_ask.add_argument("--top-k", type=int, default=3, help="Number of context chunks to use")

    # trace
    p_trace = subparsers.add_parser("trace", help="Inspect distributed telemetry traces")
    trace_sub = p_trace.add_subparsers(dest="trace_cmd", required=True)
    p_trace_list = trace_sub.add_parser("list", help="List recent traces")
    p_trace_list.add_argument("--limit", type=int, default=20, help="Max traces to show")
    p_trace_list.add_argument("--status", choices=["OK", "ERROR"], help="Filter by status")

    p_trace_show = trace_sub.add_parser("show", help="Show trace waterfall details")
    p_trace_show.add_argument("trace_id", help="Trace ID to inspect")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run automated RAG Triad benchmark suite")
    p_eval.add_argument("--threshold", type=float, default=0.75, help="Passing quality threshold (0.0 - 1.0)")

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Start the Observability Web UI")
    p_dash.add_argument("--port", type=int, default=8000, help="Port to run web dashboard")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "trace":
        if args.trace_cmd == "list":
            cmd_trace_list(args)
        elif args.trace_cmd == "show":
            cmd_trace_show(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)


if __name__ == "__main__":
    main()
