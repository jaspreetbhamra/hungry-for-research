from __future__ import annotations

import argparse

from chains.hybrid_qa import HybridQA
from common.config import secrets, yaml_config
from graph.neo4j_client import Neo4jClient
from models.llm import load_local_llm
from vectorstore.chroma_store import ChromaStore


def main():
    parser = argparse.ArgumentParser(description="Hybrid QA: Chroma + Neo4j grounding")
    parser.add_argument("question", type=str, help="Your question")
    parser.add_argument("--collection", type=str, default="papers")
    parser.add_argument("--model", type=str, default="mistral")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--neo", action="store_true", help="Enable Neo4j grounding")
    args = parser.parse_args()

    # Load vector store + LLM
    store = ChromaStore(collection_name=args.collection)
    qa_llm = load_local_llm("llm_qa")

    # Neo4j optional
    if args.neo:
        neo = Neo4jClient(
            uri=yaml_config.neo4j.uri,
            user=secrets.neo4j_user,
            password=secrets.neo4j_password,
        )
    else:
        neo = None

    hqa = HybridQA(store=store, llm=qa_llm, neo=neo)
    result = hqa.ask(args.question)

    print("\n=== ANSWER ===\n")
    print(result["answer"])

    if result["sources"]:
        print("\n=== SOURCES ===\n")
        for s in result["sources"]:
            print(
                f"- {s['source']} (page {s.get('page')}) [chunk {s.get('chunk_index')}]"
            )

    if result["graph_facts"]:
        print("\n=== GRAPH FACTS ===\n")
        for r in result["graph_facts"]:
            print(
                f"- {r['subject']} --{r['predicate']}--> {r['object']} (paper {r['paper_id']})"
            )


if __name__ == "__main__":
    main()
