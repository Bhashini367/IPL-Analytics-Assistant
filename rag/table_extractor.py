import camelot

from langchain_core.documents import (
    Document
)


def extract_tables(
    pdf_path: str
):

    tables = camelot.read_pdf(
        pdf_path,
        pages="all"
    )

    docs = []

    for idx, table in enumerate(tables):

        df = table.df

        docs.append(
            Document(
                page_content=df.to_string(),
                metadata={
                    "source": "table",
                    "table_id": idx
                }
            )
        )

    return docs