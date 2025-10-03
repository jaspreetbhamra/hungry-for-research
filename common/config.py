from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AppConfig(BaseModel):
    data_dir: Path
    persist_dir: Path
    cache_dir: Path
    collection: str = "papers"

    max_pdf_pages: int | None = None
    timeout: int = 10
    user_agent: str = "RAG-Assistant/1.0"


class VectorStoreConfig(BaseModel):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class ChunkingConfig(BaseModel):
    mode: str = Field(default="recursive", pattern="^(recursive|sentence)$")
    chunk_size: int = 1000
    chunk_overlap: int = 200


class RetrievalConfig(BaseModel):
    mode: str = Field(default="mmr", pattern="^(similarity|mmr)$")
    k: int = 4
    fetch_k: int = 20
    lambda_mult: float = 0.5
    chain_type: str = Field(default="stuff", pattern="^(stuff|map_reduce|refine)$")


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model_name: str = "mistral"
    temperature: float = 0.2


class Neo4jYAMLConfig(BaseModel):
    enabled: bool = False
    uri: str = "bolt://localhost:7687"


class GlobalYAMLConfig(BaseModel):
    app: AppConfig
    vectorstore: VectorStoreConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    llm_qa: LLMConfig
    llm_extraction: LLMConfig
    neo4j: Neo4jYAMLConfig


def load_yaml_config(path: Path = Path("config/config.yaml")) -> GlobalYAMLConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return GlobalYAMLConfig(**raw)


class Secrets(BaseSettings):
    neo4j_user: str = Field(..., env="NEO4J_USER")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")

    class Config:
        env_file = ".env"


yaml_config = load_yaml_config()
secrets = Secrets()
