import fitz

from langchain_core.documents import (
    Document
)


def load_text_documents(
    pdf_path: str
):

    pdf = fitz.open(pdf_path)

    docs = []

    for page_num in range(len(pdf)):

        page = pdf[page_num]

        text = page.get_text("text")

        if not text.strip():
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "page": page_num + 1,
                    "source": "pdf_text"
                }
            )
        )

    pdf.close()

    return docs