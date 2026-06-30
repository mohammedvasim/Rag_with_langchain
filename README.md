# RAG Pipeline with LangChain

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline built using LangChain. It loads a local text document, processes it into a searchable format, and uses a Large Language Model (LLM) via Groq to answer questions based *only* on the provided context. A FastAPI server exposes the pipeline via a REST API.

## Features

- **Document Loading:** Reads text from a local file (`my_document.txt`).
- **Text Splitting:** Chunks the document using `RecursiveCharacterTextSplitter` (500 chars, 50 overlap) to handle LLM context window limits and improve retrieval accuracy.
- **Local Embeddings:** Generates embeddings locally using Ollama's `nomic-embed-text` model via `OllamaEmbeddings`. No external API calls are made for embedding generation.
- **Vector Database:** Uses **FAISS** to store document embeddings and efficiently retrieve the most relevant chunks.
- **LLM Integration:** Connects to **Groq** using the `llama-3.1-8b-instant` model for fast answer generation.
- **FastAPI Server:** Exposes the RAG pipeline via a `POST /query` endpoint.
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
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY="your_actual_groq_api_key_here"
   ```

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

## Project Structure

| File | Description |
|------|-------------|
| `rag.py` | Core RAG logic: document loading, chunking, embeddings, vector store, LCEL chain |
| `main.py` | FastAPI app entry point |
| `endpoints.py` | API router with `POST /query` endpoint |
| `main1.ipynb` | Jupyter notebook with alternative HuggingFace-based implementation |
| `my_document.txt` | Source document about Spider-Man |

## Pipeline Overview

1. **Environment Setup** — Loads Groq API key from `.env`
2. **Data Ingestion** — Loads `my_document.txt` via `TextLoader`
3. **Chunking** — Splits into 500-character chunks with 50-character overlap
4. **Embeddings** — Converts chunks to vectors using Ollama's `nomic-embed-text`
5. **Vector Store** — Indexes embeddings in FAISS for similarity search (top `k=5`)
6. **Prompt** — Instructs the LLM to answer using *only* the retrieved context
7. **LCEL Chain** — Connects retriever → prompt → Groq LLM → string output
8. **Response** — Returns the answer via the API
