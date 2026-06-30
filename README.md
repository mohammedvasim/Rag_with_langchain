# RAG Pipeline with LangChain

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline built using LangChain. It loads a local text document, processes it into a searchable format, and uses a Large Language Model (LLM) via Groq to answer questions based *only* on the provided context. A FastAPI server exposes the pipeline via a REST API.

## Features

- **Document Loading:** Reads text from a local file (`my_document.txt`).
- **Text Splitting:** Chunks the document using `RecursiveCharacterTextSplitter` (500 chars, 50 overlap) to handle LLM context window limits and improve retrieval accuracy.
- **Local Embeddings:** Generates embeddings locally using Ollama's `nomic-embed-text` model via `OllamaEmbeddings`. No external API calls are made for embedding generation.
- **Vector Database:** Uses **FAISS** to store document embeddings and efficiently retrieve the most relevant chunks.
- **LLM Integration:** Connects to **Groq** using the `llama-3.1-8b-instant` model for fast answer generation.
- **FastAPI Server:** Exposes the RAG pipeline via a `POST /query` endpoint.
- **LangSmith Tracing:** Every query (and all internal steps — retrieval, LLM call, parsing) is traced end-to-end and viewable in the [LangSmith dashboard](https://smith.langchain.com).
- **Jupyter Notebook:** `main1.ipynb` provides a step-by-step walkthrough of the pipeline using HuggingFace embeddings as an alternative approach.

## Prerequisites

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama (for local embeddings):**
   Install [Ollama](https://ollama.ai/) and pull the embedding model:
   ```bash
   ollama pull nomic-embed-text
   ```

3. **Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY="your_groq_api_key_here"
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY="your_langsmith_api_key_here"
   LANGSMITH_PROJECT="rag-with-langchain"
   ```
   - `GROQ_API_KEY` — from [Groq Console](https://console.groq.com)
   - `LANGSMITH_API_KEY` — from [LangSmith Settings](https://smith.langchain.com/settings)
   - `LANGSMITH_PROJECT` — optional, organises traces into a named project

## Running the Server

```bash
python main.py
```

The server starts at `http://0.0.0.0:8000`. Send a query via:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How many Spider-Man movies are there?"}'
```

## LangSmith Tracing

[LangSmith](https://smith.langchain.com) provides observability into every step of the RAG pipeline. Traces are enabled via two complementary mechanisms:

### 1. Environment-level auto-tracing (`LANGSMITH_TRACING=true`)

When `LANGSMITH_TRACING` is set, LangChain automatically wraps every component invocation (retriever, LLM, output parser) into a nested trace tree. Each query produces a trace with child spans for retrieval → prompt formatting → LLM generation → output parsing.

### 2. `@traceable` decorator (`rag.py:13,19,60`)

The `@traceable` decorator from `langsmith` is applied to the two key functions in `rag.py`:

| Function | What it traces |
|----------|---------------|
| `_init_chain()` | One-time pipeline construction (document loading → chunking → embeddings → FAISS index creation) — appears only on the **first** request due to the singleton pattern |
| `get_rag_response(query)` | Every user query end-to-end — a top-level span that wraps the `@traceable` span around the LCEL chain's auto-traced internals |

The two mechanisms nest cleanly: `get_rag_response` (decorator) → `ainvoke` (auto-tracing) → retriever / LLM / parser (auto-traced steps).

### Viewing traces

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Select your project (e.g., `rag-with-langchain`)
3. Each `POST /query` call appears as a trace — click to inspect inputs, outputs, latencies, token usage, and errors per step

## Project Structure

| File | Description |
|------|-------------|
| `rag.py` | Core RAG logic: document loading, chunking, embeddings, vector store, LCEL chain, LangSmith tracing |
| `main.py` | FastAPI app entry point |
| `endpoints.py` | API router with `POST /query` endpoint |
| `main1.ipynb` | Jupyter notebook with alternative HuggingFace-based implementation |
| `my_document.txt` | Source document about Spider-Man |

## Pipeline Overview

1. **Environment Setup** — Loads Groq API key and LangSmith config from `.env`
2. **Data Ingestion** — Loads `my_document.txt` via `TextLoader`
3. **Chunking** — Splits into 500-character chunks with 50-character overlap
4. **Embeddings** — Converts chunks to vectors using Ollama's `nomic-embed-text`
5. **Vector Store** — Indexes embeddings in FAISS for similarity search (top `k=5`)
6. **Prompt** — Instructs the LLM to answer using *only* the retrieved context
7. **LCEL Chain** — Connects retriever → prompt → Groq LLM → string output
8. **Tracing** — Every step is instrumented via `@traceable` + auto-tracing and sent to LangSmith
9. **Response** — Returns the answer via the API
