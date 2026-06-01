# Design & Implementation Documentation

## Architecture Overview

The system implements a **retrieve-then-cite RAG pipeline** for sports medicine information extraction:

```
User Query
    ↓
Dense Retrieval (FAISS) + Lexical Retrieval (BM25)
    ↓
Reciprocal Rank Fusion (RRF)
    ↓
Post-Fusion Reranking (downrank back-matter)
    ↓
LLM Answer Generation with Citations
    ↓
User Response
```

## Core Design Choices

### 1. Vector Store: FAISS
- **Why**: Lightweight, local, no external dependencies, suitable for ~200 chunks
- **Alternative**: For larger datasets (>100k chunks), would consider Pinecone/Weaviate
- **Configuration**: `allow_dangerous_deserialization=True` for serialized index loading

### 2. Framework: LangChain
- **Document handling**: `langchain_core.documents.Document` with metadata
- **Embeddings**: `HuggingFaceEmbeddings` with local caching
- **Vector store**: `FAISS.from_documents()` and `FAISS.load_local()`
- **LLM integration**: `InferenceClient` from `huggingface_hub`

### 3. Retrieval Strategy: Hybrid (Dense + Lexical)
- **Dense retrieval**: FAISS `similarity_search_with_score()`
  - Candidate count: `max(top_k, min(80, top_k * 10))` to ensure diversity
  - Uses cosine distance between normalized embeddings
  
- **Lexical retrieval**: In-process BM25
  - Matches exact medical terms ("ACL", "MCL", "tibial plateau")
  - Handles book-specific terminology
  
- **Fusion**: Reciprocal Rank Fusion (RRF)
  - Formula: `score = 1 / (k + rank)` for each retriever
  - Combines rankings: `score_dense + score_bm25`
  - Harmonic mean prevents one retriever from dominating

- **Post-Fusion Reranking**:
  - Penalty: `-0.02` for passages < 200 chars (fragments)
  - Penalty: `-0.03` for back-matter (index/bibliography pages)
  - Query term matching: +boost if query terms appear in passage

## Embedding Model Details

**Model**: `BAAI/bge-base-en-v1.5`
- **Dimensions**: 768
- **Normalized**: Yes (`encode_kwargs={"normalize_embeddings": True}`)
- **Device Strategy**: `auto` (CUDA if available, else CPU)
- **Similarity Metric**: L2 distance (works like cosine on normalized vectors)

**Why BGE-base over alternatives**:
- Better than small models but 4x lighter than large models
- Trained on 430M text pairs (medical terminology included)
- 768-dim is practical for quick FAISS indexing
- Alternative: `bge-large-en-v1.5` (1024-dim, would be 2x slower but ~2% better accuracy)

**Score Interpretation** (L2 distance on normalized embeddings):
- `d = √(2(1 - cos(θ)))` where θ is angle between vectors
- Score 0.03 = 99.95% similar (excellent match)
- Score 0.1-0.2 = 95%+ similar (good match)
- Lower scores indicate stronger semantic similarity

## Chunking Strategy

**Page-Aware Chunks**:
- No chunks cross page boundaries (preserves citation accuracy)
- Chunk size: 3600 characters (≈ 500 words)
- Overlap: 600 characters (≈ 85 words)
- Metadata per chunk: `book`, `pdf_page`, `chunk_id`, `preview`

**Why these parameters**:
- 3600 chars = enough context for medical terms without redundancy
- 600 char overlap = 17% overlap, prevents missing cross-boundary concepts
- Page boundaries = enables precise "PDF page X" citations

## Citation System

**Citation Format**:
```
Source: <book_name>, PDF page <page_number>, chunk <chunk_id>
```

**Example**:
```
Source: Clinical Sports Medicine by Brukner and Khan.pdf, PDF page 238, chunk clinical_sports_medicine_by_brukner_and_khan_0238_01
```

**Citation Matching**:
- Chunks carry `pdf_page` from original PDF extraction
- LLM is prompted not to generate citations (app adds them automatically)
- Function `_citation_lines()` deduplicates citations by book+page+chunk

## LLM Generation Pipeline

**Model**: `Qwen/Qwen2.5-7B-Instruct` (configurable via `.env`)

**Generation Strategy**:
1. **System prompt**: "You are a careful sports medicine RAG assistant. Use only the provided passages."
2. **User prompt**: Question + retrieved passages (formatted with book/page/chunk metadata)
3. **Parameters**: `temperature=0.1` (deterministic), `max_tokens=384`
4. **Post-processing**: `_with_deterministic_citations()` replaces LLM citations with verified ones

**Fallback Hierarchy**:
1. Try LLM generation
2. If token missing: Return extractive fallback (top passage + reason)
3. If generation fails: Return extractive fallback (+ exception message)
4. If no passages: Return "Not enough supporting evidence"

## Evaluation Metrics

**Metric**: `hits@5` (assignment requirement)
- Definition: Top 5 retrieved chunks contain ≥1 chunk from gold source pages
- Current performance: `5/5` (100%) on included eval set
- Evaluation questions: Manually created from PDF contents (official set not provided)

**Gold Labels Format** (`data/eval_questions.json`):
```json
{
  "question": "What are common signs and symptoms of acute concussion?",
  "gold_book": "Clinical Sports Medicine by Brukner and Khan.pdf",
  "gold_pages": [238, 239, 242]
}
```

## Module Structure

```
rag_app/
├── __main__.py      # CLI entry point, delegates to cli.py
├── cli.py          # Typer commands (build-index, ask, evaluate, ui)
├── config.py       # Settings from .env (load_settings())
├── llm.py          # LLM calls, answer generation, citations
├── pipeline.py     # RAGPipeline class, retrieval, reranking
├── vector_store.py # FAISS index operations, embeddings
├── pdf.py          # PDF extraction, text+OCR
├── chunking.py     # Page-aware text splitting
├── ui.py           # Gradio web interface
└── evaluation.py   # Hits@5 metric calculation
```

## Performance Characteristics

**Index Building**:
- ~100 chunks from Grant Cooper (~20 pages): ~30 seconds
- ~2000 chunks from Brukner-Khan (~1000 pages): ~5 minutes
- Embedding: 0.5-1 MB per 1000 chunks

**Query Latency**:
- Retrieval (FAISS + BM25 + RRF): ~100 ms
- LLM generation: ~2-5 seconds (depends on model/API)
- Total round-trip: ~2.5-5.5 seconds

**Memory Usage**:
- FAISS index: ~200 MB (199 chunks × 768 dims × 4 bytes)
- Loaded embeddings model: ~350 MB
- Total: ~550 MB + Python overhead

## Known Limitations & Future Work

**Current Limitations**:
- LLM availability depends on HuggingFace account/token
- Full indexing takes time (1000+ page PDF)
- Evaluation limited to 5 questions (manual set)
- No chapter/section detection for hierarchical navigation

**Potential Improvements**:
1. **Cross-encoder reranker**: Use more sophisticated model (e.g., `ms-marco-MiniLM-L-12-v2`) to rerank top-k before LLM
2. **Chapter detection**: Extract document structure for better context
3. **Cached extraction**: Store extracted pages as JSONL to skip re-extraction
4. **Citation verifier**: Validate that cited chunks were actually retrieved
5. **Domain embeddings**: Use biomedical models (`BioBERT`, `PubMedBERT`)
6. **Stronger evaluation**: Expand to 50+ questions with inter-annotator agreement
7. **Web scraping**: Index additional sports medicine sources beyond the 2 PDFs

## Design Trade-offs

| Decision | Choice | Alternative | Trade-off |
|----------|--------|-------------|-----------|
| Vector Store | FAISS | Pinecone | Local vs Cloud (scalability) |
| Hybrid Retrieval | Dense + BM25 | Dense only | Complexity vs accuracy |
| Embedding Model | BGE-base (768) | BGE-large (1024) | Speed vs accuracy |
| Chunk Size | 3600 chars | Smaller/larger | Context vs precision |
| LLM Backend | HuggingFace | OpenAI/Anthropic | Cost vs proprietary |
| Scoring | L2 distance | Cosine similarity | Normalized embedding requirement |
