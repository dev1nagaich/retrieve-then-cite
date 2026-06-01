from __future__ import annotations

import json
from pathlib import Path

from rag_app.pipeline import RAGPipeline


def load_eval_questions(eval_path: Path) -> list[dict]:
    with eval_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Evaluation file must contain a list of question objects.")
    return data


def _is_hit(retrieved: list[dict], gold_sources: list[dict]) -> bool:
    for item in retrieved:
        for gold in gold_sources:
            same_book = item.get("book") == gold.get("book")
            page = int(item.get("pdf_page"))
            gold_pages = gold.get("pdf_pages") or [gold.get("pdf_page")]
            if same_book and page in gold_pages:
                return True
    return False


def run_evaluation(
    *,
    pipeline: RAGPipeline,
    eval_path: Path,
    top_k: int,
    example_answers: int = 2,
    use_llm: bool = True,
) -> dict:
    questions = load_eval_questions(eval_path)
    results: list[dict] = []
    hits = 0

    for index, item in enumerate(questions):
        question = item["question"]
        answer_result = pipeline.answer(
            question,
            top_k=top_k,
            use_llm=use_llm and index < example_answers,
        )
        hit = _is_hit(answer_result["retrieved"], item["gold_sources"])
        hits += int(hit)
        results.append(
            {
                "id": item.get("id"),
                "question": question,
                "hit": hit,
                "gold_sources": item["gold_sources"],
                "retrieved": answer_result["retrieved"],
                "answer": answer_result["answer"] if index < example_answers else None,
            }
        )

    return {
        "top_k": top_k,
        "hits_at_k": hits,
        "total": len(questions),
        "accuracy": hits / len(questions) if questions else 0.0,
        "results": results,
    }
