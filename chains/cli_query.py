from __future__ import annotations

import argparse

from chains.qa_chain_builder import build_qa_chain, format_sources
from common.logger import get_logger
from models.llm import load_local_llm
from retrieval.filters import build_where_filter
from retrieval.retriever_factory import build_retriever
from vectorstore.chroma_store import ChromaStore

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Query Chroma-backed RAG with retriever and chain switches."
    )
    parser.add_argument("--collection", type=str, default="papers")
    parser.add_argument(
        "--mode", type=str, default="similarity", choices=["similarity", "mmr"]
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--fetch_k", type=int, default=20)
    parser.add_argument("--lambda_mult", type=float, default=0.5)
    parser.add_argument(
        "--chain_type",
        type=str,
        default="stuff",
        choices=["stuff", "map_reduce", "refine"],
    )
    parser.add_argument("--model", type=str, default="mistral")  # Ollama model
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--include_sources",
        nargs="*",
        help="Filter: restrict to these sources (filenames/URLs)",
    )
    parser.add_argument(
        "--types", nargs="*", help="Filter: doc types e.g. pdf text web"
    )
    parser.add_argument("--min_page", type=int, default=None)
    parser.add_argument("--max_page", type=int, default=None)
    parser.add_argument("question", type=str, help="Your question")
    args = parser.parse_args()

    # Vector store (persisted)
    store = ChromaStore(collection_name=args.collection)

    # Filters
    where = build_where_filter(
        sources=args.include_sources,
        types=args.types,
        min_page=args.min_page,
        max_page=args.max_page,
    )

    # Retriever
    retriever = build_retriever(
        store=store,
        mode=args.mode,
        k=args.k,
        fetch_k=args.fetch_k,
        lambda_mult=args.lambda_mult,
        where=where if where else None,
    )

    # LLM
    qa_llm = load_local_llm("llm_qa")

    # Chain
    chain = build_qa_chain(
        llm=qa_llm,
        retriever=retriever,
        chain_type=args.chain_type,
        return_source_documents=True,
    )

    # Ask
    result = chain.invoke({"query": args.question})
    answer = result["result"] if "result" in result else result.get("answer", "")
    sources = result.get("source_documents", [])

    print("\n=== ANSWER ===\n")
    print(answer.strip())

    if sources:
        print("\n=== SOURCES ===\n")
        for meta in format_sources(sources):
            print(
                f"- {meta['source']} (page {meta.get('page')}) [chunk {meta.get('chunk_index')}]"
            )
            print(f"  snippet: {meta['snippet']}\n")


if __name__ == "__main__":
    main()
