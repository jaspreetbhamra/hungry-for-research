from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import orjson
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from common.logger import get_logger
from common.settings import settings
from ingestion.chunkers import chunk_documents
from ingestion.document_models import Chunk, RawDoc
from ingestion.loaders import load_from_path, load_from_urls_file
from vectorstore.chroma_store import ChromaStore

log = get_logger(__name__)


def discover_files(root: Path) -> List[Path]:
    paths: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in settings.allowed_exts:
            paths.append(p)
    return sorted(paths)


def _dedup_chunks(chunks: Iterable[Chunk]) -> List[Chunk]:
    seen = set()
    uniq: List[Chunk] = []
    for c in chunks:
        key = c.content_sha1
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq


@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
def _upsert_with_retry(store: ChromaStore, chunks: List[Chunk]) -> int:
    return store.upsert_chunks(chunks)


def ingest_folder(
    input_dir: Path,
    urls_file: Path | None = None,
    collection_name: str = "papers",
) -> None:
    store = ChromaStore(collection_name=collection_name)

    # 1) Load local files
    files = discover_files(input_dir)
    log.info("Discovered %d files", len(files))

    raw_docs: List[RawDoc] = []
    for f in tqdm(files, desc="Loading files"):
        for d in load_from_path(f):
            raw_docs.append(d)

    # 2) Load URLs (optional)
    if urls_file and urls_file.exists():
        log.info("Loading URLs from %s", urls_file)
        for d in tqdm(load_from_urls_file(urls_file), desc="Loading URLs"):
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
    out = settings.cache_dir / f"manifest_{collection_name}.json"
    out.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
    log.info("Wrote manifest to %s", out)
