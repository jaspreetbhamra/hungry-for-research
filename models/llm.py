from __future__ import annotations

from langchain_ollama import OllamaLLM

from common.config import yaml_config
from common.logger import get_logger

log = get_logger(__name__)


def load_local_llm(config_section="llm_qa"):
    """
    Load an LLM based on config section (llm_qa or llm_extraction).
    """
    cfg = getattr(yaml_config, config_section)

    if cfg.provider == "ollama":
        return OllamaLLM(model=cfg.model_name, temperature=cfg.temperature)
    else:
        raise ValueError(f"Unsupported provider: {cfg.provider}")
