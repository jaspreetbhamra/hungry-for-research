from __future__ import annotations

import argparse

from chains.hybrid_answerer import HybridAnswerer
from common.config import yaml_config
from graph.neo4j_client import Neo4jClient
from models.llm import load_local_llm
from vectorstore.chroma_store import ChromaStore


def main():
    p = argparse.ArgumentParser("Hybrid QA: Chroma + Neo4j")
    p.add_argument("question", type=str)
    p.add_argument("--collection", type=str, default=yaml_config.app.collection)
    p.add_argument("--mode", choices=["similarity", "mmr"], default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--fetch_k", type=int, default=None)
    p.add_argument("--lambda_mult", type=float, default=None)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--max_text_chars", type=int, default=8000)
    p.add_argument("--max_graph_lines", type=int, default=30)
    args = p.parse_args()

    store = ChromaStore(collection_name=args.collection)
    qa_llm = load_local_llm("llm_qa")  # uses your llm_qa config
    neo = Neo4jClient() if (yaml_config.neo4j.enabled and not args.no_graph) else None

    h = HybridAnswerer(store=store, qa_llm=qa_llm, neo=neo)
    result = h.ask(
        args.question,
        mode=args.mode,
        k=args.k,
        fetch_k=args.fetch_k,
        lambda_mult=args.lambda_mult,
        graph_enabled=not args.no_graph,
        max_text_chars=args.max_text_chars,
        max_graph_lines=args.max_graph_lines,
    )

    print("\n=== ANSWER ===\n")
    print(result.answer)

    if result.sources:
        print("\n=== SOURCES ===\n")
        for s in result.sources:
            print(
                f"- {s['source']} (page {s.get('page')}) [chunk {s.get('chunk_index')}]"
            )

    if result.graph_facts:
        print("\n=== GRAPH FACTS ===\n")
        for r in result.graph_facts:
            print(
                f"- {r['subject']} --{r['predicate']}--> {r['object']} (paper {r['paper_id']})"
            )


if __name__ == "__main__":
    main()


# Concepts:
# Config-driven defaults with explicit override args on ask()
# Retries for Neo4j lookups, robust logging
# Context budgeting (max_text_chars, max_graph_lines) to avoid prompt blowups
# Clear separation of concerns (retrieval, graph fetch, prompt build, answer)
# Streamlit-ready: everything is pure Python + args; easy to surface sliders/toggles for k, mode, graph_enabled, etc.
