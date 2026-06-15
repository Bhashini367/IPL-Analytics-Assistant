import streamlit as st

from graph.workflow import app_graph

st.title(
    "IPL LangGraph Multimodal RAG"
)

query = st.text_input(
    "Ask a Question"
)

if st.button("Submit"):

    result = app_graph.invoke(
        {
            "query": query
        }
    )

    st.write(
        result["answer"]
    )