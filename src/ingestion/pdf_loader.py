"""Utilities for loading and chunking PDF documents."""
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200


def load_pdf_chunks(pdf_path: str | Path) -> List[Document]:
    """Load a PDF file and split the pages into overlapping text chunks.

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        A list of LangChain ``Document`` chunks ready for downstream processing.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)
