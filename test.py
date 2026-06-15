from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

db = FAISS.load_local(
    "vector_store",
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ),
    allow_dangerous_deserialization=True
)

docs = db.similarity_search(
    "Virat Kohli runs",
    k=5
)

for i,d in enumerate(docs):
    print("="*50)
    print(i)
    print(d.page_content[:1500])
    print(d.metadata)