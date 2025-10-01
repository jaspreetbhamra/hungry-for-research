"""Factories for FAISS-based vector stores."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path("models") / "embeddings"


def build_faiss_store(
    documents: Sequence[Document],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str | None = None,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
) -> FAISS:
    """Create a FAISS vector store from a collection of LangChain documents.

    Args:
        documents: Iterable of ``Document`` objects to index. Must not be empty.
        model_name: Sentence-transformers model identifier to use for embeddings.
        device: Optional device string (e.g. ``"cpu"`` or ``"cuda"``).

    Returns:
        An in-memory FAISS vector store containing the embedded documents.
    """
    if not documents:
        raise ValueError("documents must not be empty")

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    embeddings = SentenceTransformerEmbeddings(
        model_name=model_name,
        device=device,
        cache_folder=str(cache_dir) if cache_dir is not None else None,
    )
    return FAISS.from_documents(list(documents), embedding=embeddings)
