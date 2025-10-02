from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Directories
    data_dir: Path = Field(default=Path("data"))
    persist_dir: Path = Field(default=Path("data/chroma"))
    cache_dir: Path = Field(default=Path("data/cache"))

    # Embeddings
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    splitter_mode: str = "recursive"  # "recursive" | "sentence"

    # Ingestion
    max_pdf_pages: int | None = None  # None = all
    allowed_exts: tuple[str, ...] = (".pdf", ".txt", ".md")

    # Web loader
    user_agent: str = "PersonalResearchAssistant/1.0 (+https://local.app)"
    timeout: int = 20

    class Config:
        env_file = ".env"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.persist_dir.mkdir(parents=True, exist_ok=True)
settings.cache_dir.mkdir(parents=True, exist_ok=True)
