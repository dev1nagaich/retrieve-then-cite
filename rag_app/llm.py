from __future__ import annotations

from langchain_core.documents import Document
from huggingface_hub import InferenceClient

from rag_app.config import Settings


def _format_context(docs: list[Document]) -> str:
    blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] Book: {meta.get('book')}",
                    f"PDF page: {meta.get('pdf_page')}",
                    f"Chunk: {meta.get('chunk_id')}",
                    f"Passage: {doc.page_content}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _extractive_fallback(question: str, docs: list[Document], reason: str) -> str:
    if not docs:
        return "Answer: I could not find enough supporting evidence in the indexed books.\n\nCitations: none"

    top = docs[0]
    meta = top.metadata
    excerpt = " ".join(top.page_content.split())[:900]
    return (
        "Answer: The LLM answer step was not available, so this is an extractive "
        "evidence summary from the top retrieved passage. "
        f"For the question `{question}`, the most relevant passage says: {excerpt}\n\n"
        "Citations:\n"
        f"- Source: {meta.get('book')}, PDF page {meta.get('pdf_page')}, "
        f"chunk {meta.get('chunk_id')}\n\n"
        f"LLM status: {reason}"
    )


def _citation_lines(docs: list[Document]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        meta = doc.metadata
        line = (
            f"- Source: {meta.get('book')}, PDF page {meta.get('pdf_page')}, "
            f"chunk {meta.get('chunk_id')}"
        )
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return lines


def _with_deterministic_citations(answer_text: str, docs: list[Document]) -> str:
    answer = answer_text.split("Citations:", 1)[0].strip()
    if not answer.lower().startswith("answer:"):
        answer = f"Answer: {answer}"

    citations = "\n".join(_citation_lines(docs))
    return f"{answer}\n\nCitations:\n{citations}"


def generate_answer(
    *,
    question: str,
    docs: list[Document],
    settings: Settings,
    use_llm: bool = True,
) -> str:
    """Generate a cited answer using Hugging Face, with a deterministic fallback."""

    if not use_llm:
        return _extractive_fallback(question, docs, "LLM disabled by caller.")

    if not settings.huggingface_token:
        return _extractive_fallback(
            question,
            docs,
            "HUGGINGFACEHUB_API_TOKEN is not set in .env.",
        )

    prompt = f"""Question:
{question}

Retrieved passages:
{_format_context(docs)}

Write a concise answer using only the retrieved passages. If the passages do not
contain enough evidence, say that the books do not provide enough support.
Do not include citations; the application will attach verified citations."""

    try:
        client = InferenceClient(
            model=settings.hf_llm_repo_id,
            token=settings.huggingface_token,
        )
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful sports medicine RAG assistant. "
                        "Use only the provided passages and do not use outside knowledge."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=settings.temperature,
            max_tokens=settings.max_new_tokens,
        )
        answer_text = response.choices[0].message.content or ""
        return _with_deterministic_citations(answer_text, docs)
    except Exception as exc:  # pragma: no cover - depends on remote provider
        return _extractive_fallback(question, docs, f"Hugging Face generation failed: {exc}")
