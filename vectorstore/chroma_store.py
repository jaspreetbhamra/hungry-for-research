from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from langchain_chroma.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from common.config import yaml_config
from common.logger import get_logger
from ingestion.document_models import Chunk

log = get_logger(__name__)


class ChromaStore:
    def __init__(
        self,
        persist_dir: Path | str | None = None,
        collection_name: str | None = None,
    ):
        """
        Wrapper for Chroma vector store with HuggingFace embeddings.
        Uses config/config.yaml for defaults.
        """
        self.persist_dir = str(persist_dir or yaml_config.app.persist_dir)
        self.collection_name = collection_name or yaml_config.app.collection
        self.embeddings = HuggingFaceEmbeddings(
            model_name=yaml_config.vectorstore.embedding_model
        )
        self._db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

    @property
    def db(self) -> Chroma:
        return self._db

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        """
        Add a list of Chunk objects into Chroma.
        Deduplication should be done before calling this.
        """
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for c in chunks:
            ids.append(c.chunk_id)
            texts.append(c.text)
            metadatas.append(c.metadata | {"content_sha1": c.content_sha1})

        if not ids:
            return 0

        self._db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        log.info(
            "Upserted %d chunks into collection '%s'", len(ids), self.collection_name
        )
        return len(ids)

    def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks whose metadata.source matches the given string.
        """
        res = self._db._collection.get(
            where={"source": {"$eq": source}}, include=["ids"]
        )
        ids = res.get("ids", [])
        if ids:
            self._db._collection.delete(ids=ids)
            self._db.persist()
        log.info("Deleted %d chunks for source '%s'", len(ids), source)
        return len(ids)

    def retriever(self, k: int = 3, search_type: str = "similarity"):
        """
        Convenience method to return a retriever for ad-hoc queries.
        """
        return self._db.as_retriever(search_type=search_type, search_kwargs={"k": k})
