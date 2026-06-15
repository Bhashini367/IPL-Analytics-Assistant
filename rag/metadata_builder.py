from langchain_core.documents import Document


def enrich_metadata(doc: Document) -> Document:

    metadata = doc.metadata.copy()

    metadata["char_count"] = len(doc.page_content)

    metadata["word_count"] = len(
        doc.page_content.split()
    )

    metadata["has_numbers"] = any(
        c.isdigit()
        for c in doc.page_content
    )

    metadata["document_type"] = metadata.get(
        "source",
        "unknown"
    )

    return Document(
        page_content=doc.page_content,
        metadata=metadata
    )