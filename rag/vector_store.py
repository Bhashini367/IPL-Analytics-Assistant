from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)

from config import (
    EMBEDDING_MODEL,
    FAISS_PATH
)


def create_vectorstore(
    docs
):

    embeddings = (
        HuggingFaceEmbeddings(
            model_name=
            EMBEDDING_MODEL
        )
    )

    db = FAISS.from_documents(
        docs,
        embeddings
    )

    db.save_local(
        FAISS_PATH
    )

    return db