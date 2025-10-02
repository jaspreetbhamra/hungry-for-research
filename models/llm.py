from __future__ import annotations

from langchain_ollama import OllamaLLM

from common.logger import get_logger

log = get_logger(__name__)


def load_local_llm(model_name: str = "mistral", temperature: float = 0.2) -> OllamaLLM:
    """
    Local LLM loader via Ollama. Run `ollama pull mistral` (or your preferred model)
    beforehand. Keep temperature low for grounding to retrieved context.
    """
    log.info("Loading Ollama model: %s (T=%.2f)", model_name, temperature)
    return OllamaLLM(model=model_name, temperature=temperature)
