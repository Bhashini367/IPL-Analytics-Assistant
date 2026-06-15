from dotenv import load_dotenv

import os

load_dotenv()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

FAISS_PATH = "vector_store"

PDF_PATH = (
    "data/IPL_LangGraph_RAG_Dataset.pdf"
)