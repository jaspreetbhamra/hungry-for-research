from __future__ import annotations

from pathlib import Path

import streamlit as st

from common.config import yaml_config
from ingestion.ingest_pipeline import ingest_folder


def render_ingest_section():
    st.subheader("Ingest Documents")
    st.write("Upload papers or provide URLs to expand your local knowledge base.")

    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        accept_multiple_files=True,
        type=["pdf", "txt", "md"],
    )
    urls_text = st.text_area("Or paste URLs (one per line)")

    if st.button("Run Ingestion", type="primary"):
        with st.spinner("Running ingestion pipeline..."):
            temp_dir = Path(yaml_config.app.data_dir)
            for f in uploaded_files or []:
                path = temp_dir / f.name
                path.write_bytes(f.read())
            urls_file = None
            if urls_text.strip():
                urls_file = temp_dir / "urls.txt"
                urls_file.write_text(urls_text)
            ingest_folder(input_dir=temp_dir, urls_file=urls_file)
        st.success("✅ Ingestion complete! Data added to Chroma + Neo4j.")
