from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

_chain = None

@traceable
def _init_chain():
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_ollama import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS

    loader = TextLoader('my_document.txt')
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents_chunk = splitter.split_documents(document)

    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vector_store = FAISS.from_documents(documents_chunk, embeddings)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the context below only.
        If you don't know, say you don't know.

        context:{context}
        Question:{question}
        """
    )

    return (
        {
            "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

@traceable
async def get_rag_response(query: str):
    global _chain
    if _chain is None:
        _chain = _init_chain()
    response = await _chain.ainvoke(query)
    return response


