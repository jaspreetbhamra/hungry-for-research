from __future__ import annotations

from typing import Dict, Optional

from langchain.schema import BaseRetriever

from common.logger import get_logger
from vectorstore.chroma_store import ChromaStore

log = get_logger(__name__)


def build_retriever(
    store: ChromaStore,
    mode: str = "similarity",  # "similarity" | "mmr"
    k: int = 4,
    fetch_k: int = 20,  # used by MMR to prefetch candidates
    lambda_mult: float = 0.5,  # MMR novelty <-> relevance tradeoff
    where: Optional[Dict] = None,  # metadata filter from build_where_filter(...)
):
    """
    Return a LangChain retriever backed by Chroma with optional MMR and metadata filtering.
    """
    search_kwargs: Dict = {"k": k}
    if where:
        search_kwargs["filter"] = where

    if mode == "mmr":
        # Maximal Marginal Relevance (diversity-promoting)
        search_kwargs.update({"fetch_k": fetch_k, "lambda_mult": lambda_mult})
        retriever: BaseRetriever = store.db.as_retriever(
            search_type="mmr", search_kwargs=search_kwargs
        )
        log.info(
            "Built MMR retriever k=%d fetch_k=%d lambda=%.2f", k, fetch_k, lambda_mult
        )
    else:
        retriever = store.db.as_retriever(
            search_type="similarity", search_kwargs=search_kwargs
        )
        log.info("Built similarity retriever k=%d", k)
    return retriever
