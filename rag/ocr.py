import easyocr

from langchain_core.documents import (
    Document
)


reader = easyocr.Reader(
    ["en"],
    gpu=False
)


def image_to_documents(
    image_paths
):

    docs = []

    for path in image_paths:

        results = reader.readtext(
            path
        )

        text = "\n".join(
            r[1]
            for r in results
        )

        if not text.strip():
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": "ocr",
                    "image_path": path
                }
            )
        )

    return docs