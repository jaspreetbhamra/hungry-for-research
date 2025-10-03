from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.language_models import BaseLanguageModel

from chains.qa_chain_builder import build_qa_chain, format_sources
from graph.hybrid_context import extract_candidate_entities, format_graph_facts
from graph.neo4j_client import Neo4jClient
from retrieval.retriever_factory import build_retriever
from vectorstore.chroma_store import ChromaStore


class HybridQA:
    def __init__(
        self,
        store: ChromaStore,
        llm: BaseLanguageModel,
        neo: Optional[Neo4jClient] = None,
        retriever_mode: str = "mmr",
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        chain_type: str = "stuff",
    ):
        self.store = store
        self.llm = llm
        self.neo = neo
        self.retriever = build_retriever(
            store, mode=retriever_mode, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
        self.chain = build_qa_chain(llm, self.retriever, chain_type=chain_type)

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Run hybrid QA:
          1. Retrieve from Chroma
          2. Optionally query Neo4j for structured facts
          3. Merge both contexts
          4. Ask LLM
        """
        base = self.chain.invoke({"query": question})
        answer = base.get("result") or base.get("answer", "")
        sources = base.get("source_documents", [])

        graph_facts = []
        if self.neo:
            # extract candidate entities from retrieved docs
            ents = extract_candidate_entities(sources, top_n=5)
            if ents:
                rows = self.neo.query_entities(ents)
                if rows:
                    graph_facts = rows
                    facts_txt = format_graph_facts(rows)
                    # augment answer with structured facts section
                    answer = (
                        answer.strip()
                        + "\n\n---\nStructured facts from graph:\n"
                        + facts_txt
                    )

        return {
            "answer": answer,
            "sources": format_sources(sources),
            "graph_facts": graph_facts,
        }
