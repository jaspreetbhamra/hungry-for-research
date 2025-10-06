from __future__ import annotations

import streamlit as st

from chains.hybrid_answerer import HybridAnswerer
from common.config import yaml_config
from graph.neo4j_client import Neo4jClient
from models.llm import load_local_llm
from vectorstore.chroma_store import ChromaStore


def render_qa_section(mode: str, k: int, graph_enabled: bool):
    st.subheader("Ask a Question")

    question = st.text_input("Enter your research question:")
    ask_button = st.button("Get Answer", type="primary", disabled=not question.strip())

    if ask_button:
        with st.spinner("Retrieving + reasoning..."):
            store = ChromaStore()
            qa_llm = load_local_llm("llm_qa")
            neo = (
                Neo4jClient() if (yaml_config.neo4j.enabled and graph_enabled) else None
            )

            answerer = HybridAnswerer(store=store, qa_llm=qa_llm, neo=neo)
            result = answerer.ask(question, mode=mode, k=k, graph_enabled=graph_enabled)

        st.markdown("### 🧩 Answer")
        st.write(result.answer)

        st.markdown("### 📑 Sources")
        if result.sources:
            for s in result.sources:
                st.markdown(
                    f"- **{s['source']}** (page {s.get('page', 'N/A')}, chunk {s.get('chunk_index', '-')})"
                )
        else:
            st.info("No text sources retrieved.")

        st.markdown("### 🌐 Graph Facts")
        if result.graph_facts:
            for f in result.graph_facts:
                st.markdown(f"- {f['subject']} — **{f['predicate']}** → {f['object']}")
        else:
            st.caption("No structured facts used in this answer.")
