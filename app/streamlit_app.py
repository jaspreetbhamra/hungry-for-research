from __future__ import annotations

import streamlit as st
from components.ingest_section import render_ingest_section
from components.qa_section import render_qa_section

from common.config import yaml_config

st.set_page_config(page_title="Personalized Research Assistant", layout="wide")

st.markdown(
    """
    <style>
    .big-title { font-size:2rem; font-weight:700; margin-bottom:1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<p class='big-title'>🧠 Personalized Research Assistant</p>",
    unsafe_allow_html=True,
)
st.caption("RAG + Graph-augmented QA | Local & Private")

# --- Sidebar ---
st.sidebar.title("Settings")
st.sidebar.markdown("**Active collection:**")
st.sidebar.text(yaml_config.app.collection)
st.sidebar.divider()

mode = st.sidebar.selectbox("Retrieval mode", ["similarity", "mmr"], index=1)
k = st.sidebar.slider("Top-k Chunks", 2, 10, yaml_config.retrieval.k)
graph_enabled = st.sidebar.checkbox("Use Neo4j Graph Facts", yaml_config.neo4j.enabled)

st.sidebar.divider()
st.sidebar.caption("Environment: local")
st.sidebar.caption(f"Embedding model: {yaml_config.vectorstore.embedding_model}")

# --- Tabs ---
tab_ingest, tab_qa = st.tabs(["📚 Ingest", "💬 Ask a Question"])

with tab_ingest:
    render_ingest_section()

with tab_qa:
    render_qa_section(mode=mode, k=k, graph_enabled=graph_enabled)
