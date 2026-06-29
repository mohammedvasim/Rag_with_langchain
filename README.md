# RAG Pipeline with LangChain

This project demonstrates a simple yet powerful Retrieval-Augmented Generation (RAG) pipeline built using LangChain. It loads a local text document, processes it into a searchable format, and uses a Large Language Model (LLM) via Groq to answer questions based *only* on the provided context.

## 🚀 Features

- **Document Loading:** Reads text from a local file (`my_document.txt`).
- **Text Splitting:** Chunks the document into smaller, manageable pieces using `RecursiveCharacterTextSplitter` to handle LLM context window limits and improve retrieval accuracy.
- **Local Embeddings:** Generates embeddings locally using Hugging Face's `all-MiniLM-L6-v2` model (via `sentence-transformers`). This means no external API calls are made for embedding generation, ensuring privacy and speed.
- **Vector Database:** Uses **FAISS** (Facebook AI Similarity Search) to store the document embeddings and efficiently retrieve the most relevant chunks when a question is asked.
- **LLM Integration:** Connects to **Groq** to leverage the incredibly fast `llama3-8b-8192` model for generating answers.
- **LCEL Chain:** Uses the modern LangChain Expression Language (LCEL) to build the retrieval and QA chain seamlessly.

## 🛠️ Prerequisites

Before running the code, ensure you have the necessary dependencies installed and your environment variables set up.

1. **Install Dependencies:**
   Make sure your virtual environment is activated, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables:**
   You need a Groq API key to use their LLM. Create a `.env` file in the root directory and add your key:
   ```env
   GROQ_API_KEY="your_actual_groq_api_key_here"
   ```

## 🏗️ Pipeline Overview (How it works)

Here is a step-by-step breakdown of the flow in the Jupyter Notebook (`main1.ipynb`):

1. **Environment Setup:** Loads the Groq API key from the `.env` file securely.
2. **Data Ingestion:** Loads the source material (`my_document.txt`) using LangChain's `TextLoader`.
3. **Chunking:** Splits the loaded document into 500-character chunks with a 50-character overlap. The overlap ensures that context isn't lost if a sentence is split across two chunks.
4. **Vector Store Creation:** 
   - Initializes `HuggingFaceEmbeddings` to convert text chunks into numerical vectors.
   - Creates a FAISS vector store to index these embeddings for fast searching.
5. **Retriever Setup:** Configures the FAISS index as a retriever to fetch the top `k=5` most similar chunks when given a query.
6. **LLM & Prompt Formulation:** 
   - Initializes the `ChatGroq` model.
   - Sets up a `ChatPromptTemplate` that explicitly instructs the LLM to answer the user's question using *only* the retrieved context.
7. **Chain Construction (LCEL):** 
   - Connects the retriever to fetch context based on the user's input.
   - Formats the retrieved documents and passes them along with the question into the prompt.
   - Feeds the prompt to the Groq LLM.
   - Parses the output into a clean string using `StrOutputParser`.
8. **Execution:** Invokes the chain with a user query (e.g., *"How many spiderman movies are there till date?"*) and prints the synthesized response.
