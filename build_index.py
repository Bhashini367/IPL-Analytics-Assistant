from config import PDF_PATH

from rag.loader import (
    load_text_documents
)

from rag.table_extractor import (
    extract_tables
)

from rag.chunk_builder import (
    table_to_chunks
)

from rag.metadata_builder import (
    enrich_metadata
)

from rag.vector_store import (
    create_vectorstore
)


print(
    "Loading text..."
)

text_docs = (
    load_text_documents(
        PDF_PATH
    )
)

print(
    "Loading tables..."
)

table_docs = (
    extract_tables(
        PDF_PATH
    )
)

all_docs = []

for table in table_docs:

    chunks = table_to_chunks(
        table
    )

    all_docs.extend(
        chunks
    )

all_docs.extend(
    text_docs
)

final_docs = [
    enrich_metadata(doc)
    for doc in all_docs
]

print(
    "Total Chunks:",
    len(final_docs)
)

create_vectorstore(
    final_docs
)

print(
    "FAISS Created"
)