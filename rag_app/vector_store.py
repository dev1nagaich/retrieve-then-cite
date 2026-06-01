from __future__ import annotations

import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console
from tqdm import tqdm

from rag_app.chunking import chunk_pages
from rag_app.config import Settings
from rag_app.pdf import discover_pdfs, extract_pdf_pages


console = Console()


def make_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    device = _resolve_embedding_device(settings.embedding_device)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _resolve_embedding_device(device_setting: str) -> str:
    if device_setting and device_setting != "auto":
        return device_setting

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_index(
    *,
    books_dir: Path,
    index_dir: Path,
    settings: Settings,
    chunk_size: int = 3600,
    chunk_overlap: int = 600,
    force: bool = False,
    max_pages: int | None = None,
) -> int:
    if index_dir.exists():
        if force:
            shutil.rmtree(index_dir)
        else:
            raise FileExistsError(
                f"Index already exists at {index_dir}. Use --force to rebuild it."
            )

    pdfs = discover_pdfs(books_dir)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {books_dir}")

    all_docs: list[Document] = []
    for pdf_path in tqdm(pdfs, desc="Extracting PDFs"):
        pages, warnings = extract_pdf_pages(pdf_path, max_pages=max_pages)
        for warning in warnings[:10]:
            console.print(f"[yellow]Warning:[/] {warning}")
        if len(warnings) > 10:
            console.print(f"[yellow]Warning:[/] {len(warnings) - 10} more OCR warnings.")

        docs = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_docs.extend(docs)
        console.print(f"[green]Loaded[/] {len(docs)} chunks from {pdf_path.name}")

    if not all_docs:
        raise ValueError("No text chunks were created from the PDFs.")

    embeddings = make_embeddings(settings)
    vector_store = FAISS.from_documents(all_docs, embeddings)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))
    return len(all_docs)


def load_index(*, index_dir: Path, settings: Settings) -> FAISS:
    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_dir}. Run `python -m rag_app build-index` first."
        )

    embeddings = make_embeddings(settings)
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
