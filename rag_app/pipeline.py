from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from rag_app.config import DEFAULT_INDEX_DIR, Settings, load_settings
from rag_app.llm import generate_answer
from rag_app.vector_store import load_index


class RAGPipeline:
    def __init__(self, *, index_dir: Path = DEFAULT_INDEX_DIR, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.vector_store = load_index(index_dir=index_dir, settings=self.settings)
        self.documents = _docstore_documents(self.vector_store)
        self.lexical_index = build_lexical_index(self.documents)

    def retrieve(self, question: str, *, top_k: int | None = None) -> list[tuple[Document, float]]:
        k = top_k or self.settings.top_k
        candidate_k = max(k, min(80, k * 10))
        dense_candidates = self.vector_store.similarity_search_with_score(question, k=candidate_k)
        lexical_candidates = bm25_search(question, self.lexical_index, limit=candidate_k)
        fused = reciprocal_rank_fusion(
            dense_candidates=dense_candidates,
            lexical_candidates=lexical_candidates,
            top_k=candidate_k,
        )
        return rerank_candidates(question, fused)[:k]

    def answer(self, question: str, *, top_k: int | None = None, use_llm: bool = True) -> dict:
        retrieved = self.retrieve(question, top_k=top_k)
        docs = [doc for doc, _score in retrieved]
        answer = generate_answer(
            question=question,
            docs=docs,
            settings=self.settings,
            use_llm=use_llm,
        )
        return {
            "question": question,
            "answer": answer,
            "citations": [format_citation(doc) for doc in docs],
            "retrieved": [format_retrieved(doc, score) for doc, score in retrieved],
        }


def format_citation(doc: Document) -> str:
    meta = doc.metadata
    return (
        f"Source: {meta.get('book')}, PDF page {meta.get('pdf_page')}, "
        f"chunk {meta.get('chunk_id')}"
    )


def format_retrieved(doc: Document, score: float | None) -> dict:
    meta = doc.metadata
    return {
        "book": meta.get("book"),
        "pdf_page": meta.get("pdf_page"),
        "chunk_id": meta.get("chunk_id"),
        "score": None if score is None else float(score),
        "preview": meta.get("preview") or " ".join(doc.page_content.split())[:240],
    }


def rerank_candidates(
    question: str,
    candidates: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    """Apply small metadata and lexical boosts on top of RRF relevance scores."""

    query = _normalize(question)
    query_terms = {
        term
        for term in query.split()
        if len(term) > 3 and term not in {"what", "does", "with", "from", "this", "that"}
    }

    reranked: list[tuple[float, Document, float]] = []
    for doc, score in candidates:
        base_score = float(score)
        metadata_text = _normalize(
            " ".join(
                [
                    str(doc.metadata.get("book", "")),
                    str(doc.metadata.get("book_slug", "")),
                ]
            )
        )
        raw_passage_text = doc.page_content[:2000].lower()
        passage_text = _normalize(raw_passage_text)

        adjusted_score = base_score
        if len(passage_text) < 200:
            adjusted_score -= 0.02

        if _is_index_like(raw_passage_text):
            adjusted_score -= 0.04

        if _explicit_book_match(query, metadata_text):
            adjusted_score += 0.02

        if _asks_for_definition(query) and (
            "defined as" in passage_text or "definition" in passage_text
        ):
            adjusted_score += 0.015

        if query_terms:
            shared_terms = sum(1 for term in query_terms if term in passage_text)
            adjusted_score += min(shared_terms * 0.003, 0.03)

        reranked.append((adjusted_score, doc, base_score))

    reranked.sort(key=lambda item: item[0], reverse=True)
    return [(doc, base_score) for _adjusted, doc, base_score in reranked]


@dataclass(frozen=True)
class LexicalIndex:
    documents: list[Document]
    token_counts: list[Counter[str]]
    doc_lengths: list[int]
    document_frequency: Counter[str]
    avg_doc_length: float


def build_lexical_index(documents: list[Document]) -> LexicalIndex:
    token_counts: list[Counter[str]] = []
    doc_lengths: list[int] = []
    document_frequency: Counter[str] = Counter()

    for doc in documents:
        tokens = _tokenize_document(doc)
        counts = Counter(tokens)
        token_counts.append(counts)
        doc_lengths.append(len(tokens))
        document_frequency.update(counts.keys())

    avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    return LexicalIndex(
        documents=documents,
        token_counts=token_counts,
        doc_lengths=doc_lengths,
        document_frequency=document_frequency,
        avg_doc_length=avg_doc_length,
    )


def bm25_search(
    question: str,
    index: LexicalIndex,
    *,
    limit: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[Document, float]]:
    query_terms = _tokenize(question)
    if not query_terms or not index.documents:
        return []

    total_docs = len(index.documents)
    scores: list[tuple[Document, float]] = []
    for doc, counts, doc_length in zip(index.documents, index.token_counts, index.doc_lengths):
        score = 0.0
        for term in query_terms:
            term_frequency = counts.get(term, 0)
            if term_frequency == 0:
                continue

            doc_frequency = index.document_frequency[term]
            idf = math.log(1 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
            denominator = term_frequency + k1 * (
                1 - b + b * doc_length / max(index.avg_doc_length, 1)
            )
            score += idf * (term_frequency * (k1 + 1)) / denominator

        if score > 0:
            scores.append((doc, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:limit]


def reciprocal_rank_fusion(
    *,
    dense_candidates: list[tuple[Document, float]],
    lexical_candidates: list[tuple[Document, float]],
    top_k: int,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> list[tuple[Document, float]]:
    fused_scores: defaultdict[str, float] = defaultdict(float)
    docs_by_key: dict[str, Document] = {}

    for rank, (doc, _score) in enumerate(dense_candidates, start=1):
        key = _chunk_key(doc)
        docs_by_key[key] = doc
        fused_scores[key] += dense_weight / (rrf_k + rank)

    for rank, (doc, _score) in enumerate(lexical_candidates, start=1):
        key = _chunk_key(doc)
        docs_by_key[key] = doc
        fused_scores[key] += lexical_weight / (rrf_k + rank)

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [(docs_by_key[key], score) for key, score in ranked[:top_k]]


def _docstore_documents(vector_store) -> list[Document]:
    docstore_dict = getattr(vector_store.docstore, "_dict", {})
    return list(docstore_dict.values())


def _chunk_key(doc: Document) -> str:
    return str(doc.metadata.get("chunk_id") or doc.metadata.get("source_id") or id(doc))


def _tokenize_document(doc: Document) -> list[str]:
    metadata = doc.metadata
    text = " ".join(
        [
            str(metadata.get("book", "")),
            str(metadata.get("book_slug", "")),
            doc.page_content,
        ]
    )
    return _tokenize(text)


def _tokenize(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 2 and token not in _STOPWORDS
    ]


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _explicit_book_match(query: str, metadata_text: str) -> bool:
    essential_query = "essential sports medicine" in query or "grant cooper" in query
    essential_book = "essntial sports medicine" in metadata_text or "grant cooper" in metadata_text
    clinical_query = (
        "clinical sports medicine" in query
        or "brukner" in query
        or "khan" in query
    )
    clinical_book = (
        "clinical sports medicine" in metadata_text
        or "brukner" in metadata_text
        or "khan" in metadata_text
    )
    return (essential_query and essential_book) or (clinical_query and clinical_book)


def _asks_for_definition(query: str) -> bool:
    return any(term in query for term in ("define", "defined", "definition"))


def _is_index_like(passage_text: str) -> bool:
    comma_page_refs = len(re.findall(r",\s*\d{2,4}", passage_text))
    see_refs = passage_text.count(" see ")
    return " index " in f" {passage_text[:200]} " or (comma_page_refs >= 8 and see_refs >= 1)


_STOPWORDS = {
    "the",
    "and",
    "for",
    "are",
    "with",
    "what",
    "when",
    "where",
    "which",
    "who",
    "how",
    "does",
    "did",
    "from",
    "into",
    "about",
    "this",
    "that",
    "than",
    "then",
    "also",
    "such",
}
