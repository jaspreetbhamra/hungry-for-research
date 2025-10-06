from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain.docstore.document import Document
from langchain_core.language_models import BaseLanguageModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from common.config import yaml_config
from common.logger import get_logger
from graph.hybrid_context import extract_candidate_entities, format_graph_facts
from graph.neo4j_client import Neo4jClient
from retrieval.retriever_factory import build_retriever
from vectorstore.chroma_store import ChromaStore

log = get_logger(__name__)


@dataclass(frozen=True)
class HybridAnswer:
    answer: str
    sources: List[Dict[str, Any]]
    graph_facts: List[Dict[str, Any]]


def _format_sources(docs: Sequence[Document]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in docs:
        out.append(
            {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "chunk_index": d.metadata.get("chunk_index"),
                "source_id": d.metadata.get("source_id"),
                "snippet": d.page_content[:300],
            }
        )
    return out


def _trim_context(
    docs: Sequence[Document], max_chars: int
) -> Tuple[str, List[Document]]:
    """
    Concatenate doc contents up to a character budget, preserving order.
    Returns (context_string, docs_used).
    """
    used: List[Document] = []
    buff: List[str] = []
    remaining = max_chars
    for d in docs:
        text = d.page_content.strip()
        if not text:
            continue
        take = text[:remaining]
        if not take:
            break
        buff.append(take)
        used.append(d)
        remaining -= len(take)
        if remaining <= 0:
            break
    return "\n\n---\n\n".join(buff), used


def _build_hybrid_prompt(question: str, text_context: str, graph_context: str) -> str:
    """
    Single-pass hybrid prompt that includes both retrieved text and structured facts.
    Citations are requested explicitly.
    """
    return (
        "You are a precise research assistant. Use ONLY the information provided below.\n"
        "If the answer is not completely supported by the information, say you don't know.\n"
        "When you use a text snippet, cite it like (source: <source>, page: <page>). "
        "When you use a structured fact, cite it as (graph).\n\n"
        f"Question:\n{question}\n\n"
        "Text context:\n"
        f"{text_context}\n\n"
        "Structured facts (graph):\n"
        f"{graph_context}\n\n"
        "Answer succinctly with citations:"
    )


class HybridAnswerer:
    """
    Hybrid QA that:
      1) Retrieves text chunks from Chroma
      2) Optionally queries Neo4j for structured facts related to retrieved entities
      3) Builds a single hybrid prompt and asks the QA LLM to answer with citations

    Defaults come from config/config.yaml, but explicit method args override.
    """

    def __init__(
        self,
        store: Optional[ChromaStore] = None,
        qa_llm: Optional[BaseLanguageModel] = None,
        neo: Optional[Neo4jClient] = None,
    ):
        self.store = store or ChromaStore()
        self.qa_llm = (
            qa_llm  # pass your QA LLM instance (e.g., load_local_llm("llm_qa"))
        )
        self.neo = neo
        self._graph_enabled_default = yaml_config.neo4j.enabled

        # sensible defaults from config (can be overridden in ask())
        self._retr_mode = yaml_config.retrieval.mode
        self._k = yaml_config.retrieval.k
        self._fetch_k = yaml_config.retrieval.fetch_k
        self._lambda_mult = yaml_config.retrieval.lambda_mult

    @retry(
        reraise=True,
        wait=wait_exponential(multiplier=1, min=0.5, max=6),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
    )
    def _neo_query_entities(self, entities: List[str]) -> List[Dict[str, Any]]:
        if not self.neo or not entities:
            return []
        try:
            return self.neo.query_entities(entities)
        except Exception as e:
            log.warning("Neo4j query failed (will retry): %s", e)
            raise

    def ask(
        self,
        question: str,
        *,
        # retrieval overrides
        mode: Optional[str] = None,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
        # graph overrides
        graph_enabled: Optional[bool] = None,
        graph_entities_top_n: int = 6,
        # context + answer shaping
        max_text_chars: int = 8000,
        max_graph_lines: int = 30,
    ) -> HybridAnswer:
        """
        Execute a hybrid QA round trip and return the final answer + sources + graph facts.

        Args override config when provided.
        """
        if not self.qa_llm:
            raise ValueError("HybridAnswerer requires a QA LLM instance (qa_llm).")

        # 1) Build retriever
        retriever = build_retriever(
            store=self.store,
            mode=mode or self._retr_mode,
            k=k or self._k,
            fetch_k=fetch_k or self._fetch_k,
            lambda_mult=lambda_mult or self._lambda_mult,
            where=where,
        )

        # 2) Retrieve relevant docs
        # docs: List[Document] = retriever.get_relevant_documents(question)
        docs: List[Document] = retriever.invoke(question)
        if not docs:
            msg = "I don't know. No relevant context was found."
            return HybridAnswer(answer=msg, sources=[], graph_facts=[])

        # 3) (Optional) Neo4j grounding
        use_graph = (
            self._graph_enabled_default if graph_enabled is None else graph_enabled
        )
        graph_rows: List[Dict[str, Any]] = []
        graph_block = ""
        if use_graph:
            # Extract high-signal entities from retrieved docs and query the graph
            entities = extract_candidate_entities(docs, top_n=graph_entities_top_n)
            try:
                graph_rows = self._neo_query_entities(entities)
                if graph_rows:
                    graph_block = format_graph_facts(graph_rows).splitlines()[
                        :max_graph_lines
                    ]
                    graph_block = "\n".join(graph_block)
            except Exception as e:
                log.error("Neo4j grounding disabled for this query due to error: %s", e)
                graph_rows = []
                graph_block = ""

        # 4) Build hybrid context within a token/char budget
        text_context, used_docs = _trim_context(docs, max_chars=max_text_chars)

        # 5) Single-pass hybrid prompt
        prompt = _build_hybrid_prompt(
            question, text_context=text_context, graph_context=graph_block
        )

        # 6) Ask the QA LLM (with retries handled upstream by LLM client if any)
        try:
            answer = self.qa_llm.invoke(prompt)
        except Exception as e:
            log.error("QA LLM invocation failed: %s", e, exc_info=True)
            raise

        return HybridAnswer(
            answer=str(answer).strip(),
            sources=_format_sources(used_docs),
            graph_facts=graph_rows,
        )
