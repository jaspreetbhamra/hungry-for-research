from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import orjson
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from common.config import yaml_config
from common.logger import get_logger
from graph.neo4j_client import Neo4jClient
from ingestion.chunkers import chunk_documents
from ingestion.document_models import Chunk, RawDoc
from ingestion.fact_extractor_llm import (
    extract_facts_from_chunks_llm,
    upsert_facts_to_neo4j,
)
from ingestion.loaders import load_from_path, load_from_urls_file
from vectorstore.chroma_store import ChromaStore

log = get_logger(__name__)


def discover_files(root: Path) -> List[Path]:
    """
    Recursively find all supported files in the input directory.
    Supported extensions are defined in config/config.yaml (app section).
    """
    allowed_exts = (".pdf", ".txt", ".md")
    paths: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed_exts:
            paths.append(p)
    return sorted(paths)


def _dedup_chunks(chunks: Iterable[Chunk]) -> List[Chunk]:
    """
    Remove duplicate chunks based on their content SHA1.
    """
    seen = set()
    uniq: List[Chunk] = []
    for c in chunks:
        if c.content_sha1 not in seen:
            uniq.append(c)
            seen.add(c.content_sha1)
    return uniq


@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
def _upsert_with_retry(store: ChromaStore, chunks: List[Chunk]) -> int:
    """
    Retry wrapper around Chroma upserts with exponential backoff.
    """
    return store.upsert_chunks(chunks)


def ingest_folder(
    input_dir: Path | None = None,
    urls_file: Path | None = None,
    collection_name: str | None = None,
) -> None:
    """
    Ingest documents from a local folder and/or URLs file into Chroma.
    - Extracts text
    - Chunks
    - Deduplicates
    - Upserts into vectorstore
    - Extracts structured facts via LLM and upserts into Neo4j
    - Writes manifest JSON
    """
    input_dir = input_dir or yaml_config.app.data_dir
    collection_name = collection_name or yaml_config.app.collection
    store = ChromaStore(collection_name=collection_name)

    # 1) Load local files
    files = discover_files(input_dir)
    log.info("Discovered %d files", len(files))

    raw_docs: List[RawDoc] = []
    for f in tqdm(files, desc="Loading files"):
        for d in load_from_path(f):
            raw_docs.append(d)

    # 2) Load URLs (optional)
    if urls_file and Path(urls_file).exists():
        log.info("Loading URLs from %s", urls_file)
        for d in tqdm(load_from_urls_file(Path(urls_file)), desc="Loading URLs"):
            raw_docs.append(d)

    if not raw_docs:
        log.warning("No documents found to ingest.")
        return

    # 3) Chunk
    chunks = chunk_documents(raw_docs)

    # 4) Deduplicate
    chunks = _dedup_chunks(chunks)

    # 5) Upsert
    total = _upsert_with_retry(store, chunks)
    log.info("Ingest complete: %d chunks upserted into '%s'", total, collection_name)

    # 5b) Extract + upsert structured facts via LLM into Neo4j
    if yaml_config.neo4j.enabled:
        log.info("Extracting structured facts with LLM for Neo4j graph population...")
        neo = Neo4jClient()
        # with Neo4jClient() as neo:
        count = 0
        # TODO: Move the batch size and max_batches values to the config file
        for fact in extract_facts_from_chunks_llm(chunks, batch_size=5, max_batches=5):
            # for fact in extract_facts_from_chunks_llm(chunks, max_chunks=20):
            upsert_facts_to_neo4j(neo, [fact])
            count += 1
        log.info("Graph population complete: %d facts inserted", count)
        neo.close()
        log.info("Graph population complete.")

    # 6) Write manifest (for audit/debug)
    manifest = [
        {
            "chunk_id": c.chunk_id,
            "source": c.metadata.get("source"),
            "source_id": c.metadata.get("source_id"),
            "type": c.metadata.get("type"),
            "page": c.metadata.get("page"),
            "chunk_index": c.metadata.get("chunk_index"),
            "sha1": c.content_sha1,
            "len": len(c.text),
        }
        for c in chunks
    ]
    out = yaml_config.app.cache_dir / f"manifest_{collection_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
    log.info("Wrote manifest to %s", out)
