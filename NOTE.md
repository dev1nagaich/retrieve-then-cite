# One-Page Method Note

## Choices made

The prototype uses a retrieve-then-cite architecture: extract PDF text, split it into page-aware chunks, embed those chunks, store them in FAISS, retrieve the top passages for a query, and ask a Hugging Face-hosted open model to answer only from those passages.

I chose FAISS because it is lightweight, local, and appropriate for a two-book prototype. I used LangChain for document objects, text splitting, Hugging Face embeddings, FAISS integration, and LLM calls so the pipeline stays readable and easy to replace later.

Retrieval is hybrid: dense FAISS candidates are combined with a small in-process BM25 lexical retriever using reciprocal rank fusion (RRF). This improves exact medical-term and book-specific queries without replacing semantic search.

The default embedding model is `BAAI/bge-base-en-v1.5`. It offers stronger retrieval while still being practical for a 4 GB NVIDIA GPU. The app uses `EMBEDDING_DEVICE=auto`, so embeddings run on CUDA when the installed PyTorch build supports it and otherwise fall back to CPU. Embeddings are normalized so FAISS distance works like cosine similarity. If harder questions expose embedding weakness, the next upgrade would be `BAAI/bge-large-en-v1.5`, but I would only use it after checking GPU memory and indexing speed.

Chunks are page-aware and do not cross page boundaries. The default chunk size is `3600` characters with `600` characters of overlap. This keeps enough medical context for useful retrieval while preserving precise page citations.

The default retrieval count is `top_k=5`, matching the assignment's goal of measuring whether the correct source appears in the top results.

## OCR and citations

PyMuPDF handles normal page text extraction. If a page has very little text, the code attempts OCR with Tesseract through `pytesseract`. If Tesseract is unavailable, the app warns and continues.

Every chunk carries `book`, `pdf_page`, `chunk_id`, and `source_id`. Answers are prompted to cite sources in the form:

`Source: <book>, PDF page <page>, chunk <chunk_id>`

## Evaluation

The assignment mentions five test questions, but they were not included in the workspace. I created five gold-labeled questions from the provided PDFs. The metric is `hits@5`: a question is correct if one of the top five retrieved chunks matches the gold book and PDF page. The full index currently scores `5/5` hits@5 on the included evaluation set.

## What I would improve with more time

I would add a stronger cross-encoder reranker, better chapter/section detection, stronger evaluation with more questions, cached extracted-page JSONL artifacts, and a stricter citation verifier that checks whether cited chunks were actually retrieved. I would also benchmark `bge-base` against `bge-large` and domain-specific biomedical embeddings.

## Known limitations

The Hugging Face LLM depends on the selected model being available through the account/token. If the token is absent or generation fails, the app falls back to an extractive answer so retrieval and evaluation can still run. Full indexing may take time because one PDF is more than 1,000 pages.
