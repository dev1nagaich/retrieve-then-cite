# Remote Physio RAG Assignment

This is a small retrieve-then-cite RAG prototype over the two sports medicine PDFs in `books/`.

It can:

- extract text from PDFs page by page,
- OCR low-text/scanned pages when Tesseract is available,
- chunk passages with book/page metadata,
- embed chunks with a local sentence-transformer model,
- store/search vectors with FAISS,
- combine dense retrieval with BM25 lexical retrieval using reciprocal rank fusion (RRF),
- answer questions with Hugging Face through LangChain,
- run a five-question source-retrieval evaluation,
- launch a basic Gradio UI for pasted queries.

## 1. Environment setup

```powershell
conda create -n ragenv python=3.11 -y
conda activate ragenv
pip install -r requirements.txt
```

If `faiss-cpu` fails to install with pip on Windows, install it with Conda and then rerun pip:

```powershell
conda install -n ragenv -c conda-forge faiss-cpu -y
pip install -r requirements.txt
```

## 2. Configure the LLM

Put your Hugging Face token in `.env`:

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
HF_LLM_REPO_ID=HuggingFaceH4/zephyr-7b-beta
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DEVICE=auto
TOP_K=5
```

The retrieval step uses local embeddings. The Hugging Face key is only needed for generated answers. Without the key, the app still retrieves passages and returns an extractive fallback answer.

`BAAI/bge-base-en-v1.5` is the default because it gives better retrieval quality than the small model while still being reasonable for a 4 GB NVIDIA GPU. `EMBEDDING_DEVICE=auto` uses CUDA if the installed PyTorch build can see the GPU; otherwise it falls back to CPU. If you install a CUDA-enabled PyTorch build later, leave this value as `auto`.

Changing `EMBEDDING_MODEL` requires rebuilding the FAISS index because embedding dimensions can change.

## 3. Build the index

```powershell
python -m rag_app build-index --force
```

The full build processes both books, including the 1,000+ page Clinical Sports Medicine PDF, so it may take a while on CPU.

The current default index is expected to be built with `BAAI/bge-base-en-v1.5`.

## 4. Ask one question

```powershell
python -m rag_app ask "What are common signs and symptoms of acute concussion?"
```

To run retrieval without calling the LLM:

```powershell
python -m rag_app ask "What are common signs and symptoms of acute concussion?" --no-llm
```

## 5. Run evaluation

```powershell
python -m rag_app evaluate
```

This reports `hits@5`: how many of the five gold source pages appear in the top retrieved chunks. It also writes a JSON report to `artifacts/reports/evaluation_report.json`.

Current full-index retrieval result:

```text
hits@5: 5/5 (100%)
```

The retriever uses FAISS dense retrieval plus an in-process BM25 lexical retriever, then fuses both ranked lists with RRF. A small post-fusion cleanup downranks index-like back-matter pages so answer citations prefer evidence passages.

## 6. Launch the UI

```powershell
python -m rag_app ui
```

Open the local Gradio URL shown in the terminal, paste a query, choose `top_k`, and submit. The UI displays the answer, citations, and retrieved passage previews.

## OCR note

Install Tesseract OCR separately if you need scanned-page fallback:

- Windows: install Tesseract and make sure `tesseract.exe` is on `PATH`.
- If Tesseract is missing, the app warns on low-text pages and continues with normal text extraction.

## Assignment assumptions

- The official five labeled questions were not present in the workspace, so `data/eval_questions.json` contains five manually created gold-labeled questions from the provided books.
- Gold sources use PDF page numbers.
- The UI is intentionally basic because the assignment prioritizes method, citations, and measurement over frontend polish.
