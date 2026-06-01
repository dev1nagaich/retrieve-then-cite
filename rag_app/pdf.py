from __future__ import annotations

import io
import shutil
from pathlib import Path

import fitz
from PIL import Image

from rag_app.models import ExtractedPage


def _ocr_is_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


def _ocr_page(page: fitz.Page, dpi: int) -> str:
    import pytesseract

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(image)


def extract_pdf_pages(
    pdf_path: Path,
    *,
    min_text_chars: int = 80,
    ocr_dpi: int = 200,
    max_pages: int | None = None,
) -> tuple[list[ExtractedPage], list[str]]:
    """Extract page text from a PDF with OCR fallback for low-text pages."""

    pages: list[ExtractedPage] = []
    warnings: list[str] = []
    ocr_available = _ocr_is_available()

    with fitz.open(pdf_path) as doc:
        page_count = len(doc) if max_pages is None else min(max_pages, len(doc))
        for page_index in range(page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            method = "text"

            if len(text.strip()) < min_text_chars:
                if ocr_available:
                    try:
                        ocr_text = _ocr_page(page, ocr_dpi)
                        if len(ocr_text.strip()) > len(text.strip()):
                            text = ocr_text
                            method = "ocr"
                    except Exception as exc:  # pragma: no cover - depends on local OCR install
                        method = "text"
                        warnings.append(
                            f"OCR failed for {pdf_path.name} page {page_index + 1}: {exc}"
                        )
                else:
                    warnings.append(
                        f"Low text on {pdf_path.name} page {page_index + 1}; "
                        "Tesseract OCR is not installed or not on PATH."
                    )

            pages.append(
                ExtractedPage(
                    book=pdf_path.name,
                    pdf_page=page_index + 1,
                    text=text.strip(),
                    extraction_method=method,
                )
            )

    return pages, warnings


def discover_pdfs(books_dir: Path) -> list[Path]:
    return sorted(books_dir.glob("*.pdf"))
