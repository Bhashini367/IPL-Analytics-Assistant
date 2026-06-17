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


def get_retriever():

    embeddings = (
        HuggingFaceEmbeddings(
            model_name=
            EMBEDDING_MODEL
        )
    )

    db = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db.as_retriever(
        search_kwargs={
            "k": 4
        }
    )