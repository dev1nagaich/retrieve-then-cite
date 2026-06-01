from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedPage:
    book: str
    pdf_page: int
    text: str
    extraction_method: str


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float | None
    metadata: dict

    @property
    def citation(self) -> str:
        book = self.metadata.get("book", "Unknown book")
        page = self.metadata.get("pdf_page", "?")
        chunk_id = self.metadata.get("chunk_id", "?")
        return f"Source: {book}, PDF page {page}, chunk {chunk_id}"
