from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from rag_app.config import (
    DEFAULT_BOOKS_DIR,
    DEFAULT_EVAL_PATH,
    DEFAULT_INDEX_DIR,
    DEFAULT_REPORTS_DIR,
    load_settings,
)


app = typer.Typer(help="Retrieve-then-cite RAG prototype for sports medicine PDFs.")
console = Console()


@app.command("build-index")
def build_index(
    books_dir: Path = typer.Option(DEFAULT_BOOKS_DIR, help="Directory containing textbook PDFs."),
    index_dir: Path = typer.Option(DEFAULT_INDEX_DIR, help="Where to save the FAISS index."),
    chunk_size: int = typer.Option(3600, help="Character chunk size."),
    chunk_overlap: int = typer.Option(600, help="Character overlap between chunks."),
    force: bool = typer.Option(False, help="Rebuild even if an index already exists."),
    max_pages: Optional[int] = typer.Option(None, help="Debug option: limit pages per PDF."),
) -> None:
    """Extract PDFs, chunk text, embed chunks, and save a FAISS index."""

    from rag_app.vector_store import build_index as build_faiss_index

    settings = load_settings()
    count = build_faiss_index(
        books_dir=books_dir,
        index_dir=index_dir,
        settings=settings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        force=force,
        max_pages=max_pages,
    )
    console.print(f"[bold green]Saved FAISS index[/] with {count} chunks to {index_dir}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer from the indexed books."),
    index_dir: Path = typer.Option(DEFAULT_INDEX_DIR, help="Path to FAISS index."),
    top_k: Optional[int] = typer.Option(None, help="Number of chunks to retrieve."),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Skip Hugging Face generation and show extractive fallback.",
    ),
) -> None:
    """Ask one question and print a cited answer."""

    from rag_app.pipeline import RAGPipeline

    pipeline = RAGPipeline(index_dir=index_dir)
    result = pipeline.answer(question, top_k=top_k, use_llm=not no_llm)

    console.print("\n[bold]Answer[/]")
    console.print(result["answer"])
    console.print("\n[bold]Retrieved sources[/]")
    table = Table("Book", "PDF page", "Chunk", "Score", "Preview")
    for item in result["retrieved"]:
        score = "" if item["score"] is None else f"{float(item['score']):.4f}"
        table.add_row(
            str(item["book"]),
            str(item["pdf_page"]),
            str(item["chunk_id"]),
            score,
            str(item["preview"]),
        )
    console.print(table)


@app.command()
def evaluate(
    eval_path: Path = typer.Option(DEFAULT_EVAL_PATH, help="Gold-labeled evaluation questions."),
    index_dir: Path = typer.Option(DEFAULT_INDEX_DIR, help="Path to FAISS index."),
    top_k: Optional[int] = typer.Option(None, help="Number of chunks to retrieve."),
    example_answers: int = typer.Option(2, help="How many example answers to generate."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip Hugging Face generation."),
    output_path: Optional[Path] = typer.Option(None, help="Optional JSON report path."),
) -> None:
    """Run retrieval evaluation against gold source pages."""

    from rag_app.evaluation import run_evaluation
    from rag_app.pipeline import RAGPipeline

    settings = load_settings()
    k = top_k or settings.top_k
    pipeline = RAGPipeline(index_dir=index_dir, settings=settings)
    report = run_evaluation(
        pipeline=pipeline,
        eval_path=eval_path,
        top_k=k,
        example_answers=example_answers,
        use_llm=not no_llm,
    )

    console.print(
        f"[bold green]hits@{k}:[/] {report['hits_at_k']}/{report['total']} "
        f"({report['accuracy']:.0%})"
    )
    table = Table("ID", "Hit", "Question", "Top retrieved source")
    for row in report["results"]:
        top = row["retrieved"][0] if row["retrieved"] else {}
        source = f"{top.get('book')} p.{top.get('pdf_page')} {top.get('chunk_id')}"
        table.add_row(str(row["id"]), "yes" if row["hit"] else "no", row["question"], source)
    console.print(table)

    if output_path is None:
        DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_REPORTS_DIR / "evaluation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"[green]Wrote report to[/] {output_path}")


@app.command()
def ui(
    index_dir: Path = typer.Option(DEFAULT_INDEX_DIR, help="Path to FAISS index."),
    host: str = typer.Option("127.0.0.1", help="Host for the Gradio server."),
    port: int = typer.Option(7860, help="Port for the Gradio server."),
) -> None:
    """Launch the local Gradio UI."""

    from rag_app.ui import launch_ui

    launch_ui(index_dir=index_dir, host=host, port=port)
