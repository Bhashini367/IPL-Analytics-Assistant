from langchain_core.documents import Document
import pandas as pd


def table_to_chunks(table_doc):

    text = table_doc.page_content

    try:
        df = pd.read_fwf(
            pd.io.common.StringIO(text)
        )

    except Exception:
        return []

    chunks = []

    for _, row in df.iterrows():

        row_data = []

        for col in df.columns:

            value = str(row[col]).strip()

            if value and value != "nan":

                row_data.append(
                    f"{col}: {value}"
                )

        if not row_data:
            continue

        chunk_text = "\n".join(row_data)

        chunks.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "source": "table"
                }
            )
        )

    return chunks