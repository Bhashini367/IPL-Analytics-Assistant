from config import PDF_PATH

from rag.loader import load_text_documents
from rag.table_extractor import extract_tables
from rag.vector_store import create_vectorstore

print("Loading text documents...")
text_docs = load_text_documents(PDF_PATH)

print("Loading table documents...")
table_docs = extract_tables(PDF_PATH)

# Combine all documents
all_docs = table_docs + text_docs

print("Total Documents:", len(all_docs))

create_vectorstore(all_docs)

print("FAISS index created successfully.")