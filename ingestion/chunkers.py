from __future__ import annotations

from typing import Iterable, List

import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter

from common.config import yaml_config
from ingestion.document_models import Chunk, RawDoc
from ingestion.hash_utils import sha1_text

# Ensure punkt is available for sentence tokenization
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


def chunk_documents(docs: Iterable[RawDoc]) -> List[Chunk]:
    """
    Split documents into chunks according to the configured chunking mode.
    """
    if yaml_config.chunking.mode == "sentence":
        return _sentence_chunks(docs)
    else:
        return _recursive_chunks(docs)


def _recursive_chunks(docs: Iterable[RawDoc]) -> List[Chunk]:
    """
    Use LangChain's RecursiveCharacterTextSplitter to create overlapping
    character-based chunks, good for embeddings + retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=yaml_config.chunking.chunk_size,
        chunk_overlap=yaml_config.chunking.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    out: List[Chunk] = []
    for d in docs:
        for i, piece in enumerate(splitter.split_text(d.text)):
            cid = sha1_text(f"{d.source_id}::{i}::{piece[:64]}")
            out.append(
                Chunk(
                    chunk_id=cid,
                    text=piece,
                    metadata={**d.metadata, "source_id": d.source_id, "chunk_index": i},
                    content_sha1=sha1_text(piece),
                )
            )
    return out


def _sentence_chunks(docs: Iterable[RawDoc]) -> List[Chunk]:
    """
    Create chunks based on sentences, concatenated until they hit the configured max length.
    Supports overlap (carryover characters from previous chunk).
    """
    from nltk.tokenize import sent_tokenize

    out: List[Chunk] = []
    max_len = yaml_config.chunking.chunk_size
    overlap = yaml_config.chunking.chunk_overlap

    for d in docs:
        sents = sent_tokenize(d.text)
        buf: List[str] = []
        buf_len = 0
        idx = 0

        for s in sents:
            s_len = len(s)
            if buf_len + s_len + 1 <= max_len:
                buf.append(s)
                buf_len += s_len + 1
            else:
                piece = " ".join(buf).strip()
                if piece:
                    cid = sha1_text(f"{d.source_id}::{idx}::{piece[:64]}")
                    out.append(
                        Chunk(
                            chunk_id=cid,
                            text=piece,
                            metadata={
                                **d.metadata,
                                "source_id": d.source_id,
                                "chunk_index": idx,
                            },
                            content_sha1=sha1_text(piece),
                        )
                    )
                    idx += 1
                    carry = piece[-overlap:] if overlap > 0 else ""
                    buf = [carry, s] if carry else [s]
                    buf_len = len(" ".join(buf))
        # flush last buffer
        piece = " ".join(buf).strip()
        if piece:
            cid = sha1_text(f"{d.source_id}::{idx}::{piece[:64]}")
            out.append(
                Chunk(
                    chunk_id=cid,
                    text=piece,
                    metadata={
                        **d.metadata,
                        "source_id": d.source_id,
                        "chunk_index": idx,
                    },
                    content_sha1=sha1_text(piece),
                )
            )
    return out
