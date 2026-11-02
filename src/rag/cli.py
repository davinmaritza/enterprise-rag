"""Enterprise RAG Command Line Interface (CLI).

Provides commands for document ingestion, collection management,
hybrid vector/keyword search inspection, and question answering.
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
from src.rag.search.bm25_searcher import BM25Searcher
from src.rag.search.hybrid_searcher import HybridSearcher
from src.rag.search.reranker import Reranker
from src.rag.synthesizer.generator import RAGSynthesizer

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

    # 1. Hybrid Search (RRF)
    raw_hits = hybrid.search_hybrid(query=query, top_k=args.top_k * 2)

    # 2. Re-ranking
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
    """Run full RAG pipeline: retrieve, rerank, and generate grounded answer with citations."""
    query = args.query
    console.print(f"[cyan]Searching knowledge base for: [/cyan]'{query}'...")

    store = QdrantStore(url=args.url, collection_name=args.collection)
    embedder = FastEmbedder()
    hybrid = HybridSearcher(vector_store=store, embedder=embedder)
    reranker = Reranker()

    raw_hits = hybrid.search_hybrid(query=query, top_k=args.top_k * 2)
    ranked_hits = reranker.rerank(query=query, candidates=raw_hits, top_k=args.top_k)

    if not ranked_hits:
        console.print("[yellow]Tidak ditemukan konteks yang relevan dalam basis pengetahuan.[/yellow]")
        return

    synthesizer = RAGSynthesizer()
    console.print("[cyan]Synthesizing answer with strict source citations...[/cyan]\n")
    res = synthesizer.generate(query=query, context_chunks=ranked_hits)

    console.print(Panel(
        res["answer"],
        title=f"🤖 RAG Answer (Model: {res['model']})",
        subtitle=f"Used {res['used_chunks_count']} sources: {', '.join(set(res['citations']))}",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise Hybrid RAG CLI (Qdrant + BM25 + Rerank)")
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
    p_ask = subparsers.add_parser("ask", help="Ask a question with grounded citations")
    p_ask.add_argument("query", help="Question prompt")
    p_ask.add_argument("--top-k", type=int, default=3, help="Number of context chunks to use")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "ask":
        cmd_ask(args)


if __name__ == "__main__":
    main()
