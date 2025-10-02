from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RawDoc:
    source_id: str  # stable ID (path or URL)
    text: str  # full raw text
    metadata: Dict[str, Any]  # { "source": "...", "type": "pdf|txt|web", "page": ... }
    content_sha1: str  # hash for de-dup


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    content_sha1: str  # chunk-level hash
