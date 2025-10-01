"""Streamlit front end for PDF question answering."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple

import streamlit as st

from src.chains.qa_chain import build_qa_chain, load_ollama_llm
from src.ingestion.pdf_loader import load_pdf_chunks
from src.vectorstore.faiss_store import build_faiss_store
from langchain_core.language_models import BaseLanguageModel


st.set_page_config(page_title="Hungry for Research", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def get_llm(
    model_name: str,
    temperature: float,
    max_tokens: int,
    request_timeout: int,
) -> BaseLanguageModel:
    """Lazily create (and cache) an Ollama-backed LLM instance."""

    return load_ollama_llm(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
    )


def index_uploaded_pdf(file_bytes: bytes) -> Tuple[object, int]:
    """Persist the uploaded PDF temporarily and return a FAISS index plus chunk count."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = Path(tmp_file.name)

    try:
        documents = load_pdf_chunks(tmp_path)
        vectorstore = build_faiss_store(documents)
    finally:
        tmp_path.unlink(missing_ok=True)

    return vectorstore, len(documents)


st.title("Hungry for Research 🧠📄")
st.write(
    "Upload a PDF, let the app embed it locally, and query it with an Ollama-served LLaMA model."
)

with st.sidebar:
    st.header("Model Settings")
    model_name = st.text_input("Ollama model", value="llama2")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    max_tokens = st.slider("Max tokens", 64, 2048, 512, 64)
    request_timeout = st.slider("Request timeout (s)", 30, 300, 120, 10)
    top_k = st.slider("Retriever top-k", 1, 10, 4)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.doc_chunks = 0
    st.session_state.source_title = None
    st.session_state.source_signature = None

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="visible")

if uploaded_file is not None:
    file_signature = (uploaded_file.name, uploaded_file.size)
    file_changed = file_signature != st.session_state.get("source_signature")
    if file_changed or st.session_state.vectorstore is None:
        with st.spinner("Indexing PDF with local embeddings..."):
            vectorstore, num_chunks = index_uploaded_pdf(uploaded_file.getvalue())
        st.session_state.vectorstore = vectorstore
        st.session_state.doc_chunks = num_chunks
        st.session_state.source_title = uploaded_file.name
        st.session_state.source_signature = file_signature
        st.success(f"Indexed {num_chunks} chunks from {uploaded_file.name}.")

if st.session_state.vectorstore is None:
    st.info("Upload a PDF to start asking questions.")
    st.stop()

st.caption(
    f"Active index: {st.session_state.source_title} • {st.session_state.doc_chunks} chunks"
)

question = st.text_area(
    "Ask a question about the uploaded document",
    placeholder="What are the key takeaways?",
)

run_query = st.button("Run retrieval QA", type="primary", disabled=not question)

if run_query and question:
    with st.spinner("Generating answer with Ollama..."):
        llm = get_llm(model_name, temperature, max_tokens, request_timeout)
        qa_chain = build_qa_chain(
            llm,
            st.session_state.vectorstore,
            k=top_k,
            return_source_documents=True,
        )
        response = qa_chain.invoke({"query": question})
        answer = response.get("result", "No answer produced.")
        sources = response.get("source_documents", [])

    st.subheader("Answer")
    st.write(answer)

    if sources:
        st.subheader("Top sources")
        for idx, doc in enumerate(sources, start=1):
            metadata = doc.metadata or {}
            source_name = metadata.get("source", st.session_state.source_title or "Document")
            page = metadata.get("page")
            source_label = f"{source_name}"
            if page is not None:
                source_label += f" (page {page + 1})"
            st.markdown(f"**{idx}. {source_label}**")
            st.caption(doc.page_content[:400] + ("…" if len(doc.page_content) > 400 else ""))
else:
    if question:
        st.info("Press ‘Run retrieval QA’ to generate an answer.")

st.caption(
    "Ensure the Ollama daemon is running locally (`ollama serve`) and that the selected model"
    " has been pulled (`ollama pull llama2`)."
)
