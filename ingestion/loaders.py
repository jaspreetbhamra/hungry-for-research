from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from common.config import yaml_config
from common.logger import get_logger
from ingestion.cleaners import strip_header_footer
from ingestion.document_models import RawDoc
from ingestion.hash_utils import sha1_text

log = get_logger(__name__)

# Use cache directory from config
CACHE_DIR = yaml_config.app.cache_dir
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(url: str) -> str:
    """Stable hash key for a URL."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def _cache_path(url: str, kind: str) -> Path:
    """Return cache path for a given URL."""
    return CACHE_DIR / f"{_cache_key(url)}.{kind}"


def _save_cache(path: Path, data: str | bytes) -> None:
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)


def _load_cache(path: Path) -> str | bytes | None:
    if not path.exists():
        return None
    if path.suffix == ".text":
        return path.read_text(encoding="utf-8")
    else:
        return path.read_bytes()


def load_from_path(path: Path) -> Iterable[RawDoc]:
    """Load local files (.pdf, .txt, .md)."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        yield from _load_pdf(path)
    elif ext in (".txt", ".md"):
        yield _load_text_file(path)
    else:
        log.warning("Skipping unsupported file: %s", path)


def _load_text_file(path: Path) -> RawDoc:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    txt = strip_header_footer(txt)
    return RawDoc(
        source_id=str(path.resolve()),
        text=txt,
        metadata={"source": str(path.name), "type": "text"},
        content_sha1=sha1_text(txt),
    )


def load_from_urls_file(path: Path) -> Iterable[RawDoc]:
    """
    Load RawDocs from a file of URLs (skipping comments and empty lines).
    """
    with Path(path).open() as f:
        for line in f:
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            doc = load_from_url(url)
            if doc:
                yield doc


def _load_pdf(path: Path) -> Iterable[RawDoc]:
    reader = PdfReader(str(path))
    pages = reader.pages
    # Limit by config if set, else all pages
    max_pages = getattr(yaml_config.app, "max_pdf_pages", None) or len(pages)

    for i, page in enumerate(pages[:max_pages]):
        raw = page.extract_text() or ""
        cleaned = strip_header_footer(raw)
        if not cleaned.strip():
            continue
        yield RawDoc(
            source_id=f"{path.resolve()}#page={i + 1}",
            text=cleaned,
            metadata={"source": path.name, "type": "pdf", "page": i + 1},
            content_sha1=sha1_text(cleaned),
        )


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _fetch(url: str) -> requests.Response:
    """Download URL with retry logic."""
    resp = requests.get(
        url,
        timeout=getattr(yaml_config.app, "timeout", 10),
        headers={
            "User-Agent": getattr(yaml_config.app, "user_agent", "RAG-Assistant/1.0")
        },
    )
    resp.raise_for_status()
    return resp


def load_from_url(url: str) -> RawDoc | None:
    """Fetch from URL with caching, retries, and type-specific parsing."""
    text_cache = _cache_path(url, "text")
    if text_cache.exists():
        log.info("Cache hit for %s", url)
        cached = _load_cache(text_cache)
        if cached:
            return RawDoc(
                source_id=url,
                text=cached,
                metadata={"source": url, "type": "web"},
                content_sha1=sha1_text(cached),
            )

    try:
        raw_cache = _cache_path(url, "raw")

        if raw_cache.exists():
            log.info("Using cached raw for %s", url)
            content = _load_cache(raw_cache)
            headers = {}
        else:
            resp = _fetch(url)
            content = resp.content
            headers = resp.headers
            _save_cache(raw_cache, content)

        # Decide parser
        if url.lower().endswith(".pdf") or headers.get("Content-Type", "").startswith(
            "application/pdf"
        ):
            text = _extract_pdf_text(content)
        else:
            try:
                text = _extract_html_text(content.decode("utf-8", errors="ignore"))
            except Exception:
                text = str(content)

        text = strip_header_footer(text)
        if not text.strip():
            return None

        _save_cache(text_cache, text)
        return RawDoc(
            source_id=url,
            text=text,
            metadata={"source": url, "type": "web"},
            content_sha1=sha1_text(text),
        )

    except Exception as e:
        log.error("Failed to fetch %s: %s", url, e, exc_info=True)
        return None


def _extract_pdf_text(content: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(content))
        texts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(texts).strip()
    except Exception as e:
        log.error("Failed to parse PDF: %s", e, exc_info=True)
        return ""


def _extract_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return "\n".join(t.strip() for t in soup.get_text("\n").splitlines() if t.strip())
