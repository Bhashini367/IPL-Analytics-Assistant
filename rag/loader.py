import fitz
from langchain_core.documents import Document

# Pages already handled by table_extraction.py
TABLE_PAGES = {2, 3, 4, 5, 6, 7, 8, 9, 10}


def load_text_documents(pdf_path: str):
    pdf = fitz.open(pdf_path)

    docs = []

    for page_num in range(len(pdf)):
        page_number = page_num + 1

        # Skip pages whose content is already extracted as tables
        if page_number in TABLE_PAGES:
            continue

        page = pdf[page_num]
        text = page.get_text("text").strip()

        if not text:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "page": page_number,
                    "source": "pdf_text",
                    "document_type": "narrative",
                },
            )
        )

    pdf.close()

    return docs