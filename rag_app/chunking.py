from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_app.models import ExtractedPage


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return slug.strip("_")[:48] or "book"


def chunk_pages(
    pages: Iterable[ExtractedPage],
    *,
    chunk_size: int = 3600,
    chunk_overlap: int = 600,
) -> list[Document]:
    """Split page text into chunks without crossing page boundaries."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )

    docs: list[Document] = []
    for page in pages:
        if not page.text.strip():
            continue

        chunks = splitter.split_text(page.text)
        book_slug = _slug(page.book.removesuffix(".pdf"))
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_id = f"{book_slug}_{page.pdf_page:04d}_{chunk_index:02d}"
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "book": page.book,
                        "book_slug": book_slug,
                        "pdf_page": page.pdf_page,
                        "chunk_id": chunk_id,
                        "source_id": f"{page.book}#page={page.pdf_page}#chunk={chunk_index}",
                        "extraction_method": page.extraction_method,
                        "preview": " ".join(chunk.split())[:240],
                    },
                )
            )

    return docs
