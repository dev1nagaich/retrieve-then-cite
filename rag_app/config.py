from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOOKS_DIR = PROJECT_ROOT / "books"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "artifacts" / "faiss_index"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"
DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "eval_questions.json"


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    huggingface_token: str | None
    hf_llm_repo_id: str
    embedding_model: str
    embedding_device: str
    top_k: int
    temperature: float
    max_new_tokens: int


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")

    top_k_raw = os.getenv("TOP_K", "5").strip()
    try:
        top_k = max(1, int(top_k_raw))
    except ValueError:
        top_k = 5

    return Settings(
        huggingface_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") or None,
        hf_llm_repo_id=os.getenv("HF_LLM_REPO_ID", "HuggingFaceH4/zephyr-7b-beta"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "auto").strip().lower(),
        top_k=top_k,
        temperature=float(os.getenv("HF_TEMPERATURE", "0.1")),
        max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "384")),
    )
