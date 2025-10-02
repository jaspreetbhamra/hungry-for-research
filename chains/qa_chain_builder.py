from __future__ import annotations

from typing import Any, Literal

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, RetrievalQA
from langchain.chains.combine_documents.map_reduce import (
    MapReduceDocumentsChain as MRDocs,
)
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.docstore.document import Document
from langchain_core.language_models import BaseLanguageModel

from chains.prompts import (
    MAP_PROMPT,
    REDUCE_PROMPT,
    REFINE_PROMPT_QUESTION,
    STUFF_TEMPLATE,
)
from common.logger import get_logger

log = get_logger(__name__)

ChainType = Literal["stuff", "map_reduce", "refine"]


def build_qa_chain(
    llm: BaseLanguageModel,
    retriever,
    chain_type: ChainType = "stuff",
    return_source_documents: bool = True,
):
    """
    Build a QA chain with a selected composition strategy.
    """
    if chain_type == "stuff":
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": STUFF_TEMPLATE},
            return_source_documents=return_source_documents,
        )
        log.info("Initialized RetrievalQA chain (stuff).")
        return chain

    if chain_type == "map_reduce":
        # Map
        map_chain = MapReduceDocumentsChain(
            llm_chain=llm.with_prompt(MAP_PROMPT),
            document_variable_name="context",
        )
        # Reduce
        reduce_chain = ReduceDocumentsChain(
            llm_chain=llm.with_prompt(REDUCE_PROMPT),
            document_variable_name="summaries",
            collapse_documents_chain=llm.with_prompt(REDUCE_PROMPT),
        )
        mr_chain = MRDocs(
            llm_chain=map_chain.llm_chain,
            reduce_documents_chain=reduce_chain,
            document_variable_name="context",
            return_intermediate_steps=return_source_documents,
        )
        # Wrap with retriever
        from langchain.chains import RetrievalQAWithSourcesChain

        chain = RetrievalQAWithSourcesChain(
            combine_documents_chain=mr_chain,
            retriever=retriever,
            return_source_documents=return_source_documents,
        )
        log.info("Initialized QA chain (map_reduce).")
        return chain

    if chain_type == "refine":
        refine_chain = RefineDocumentsChain(
            llm_chain=llm.with_prompt(STUFF_TEMPLATE),
            refine_llm_chain=llm.with_prompt(REFINE_PROMPT_QUESTION),
            document_variable_name="context",
            initial_response_name="existing_answer",
        )
        from langchain.chains import RetrievalQAWithSourcesChain

        chain = RetrievalQAWithSourcesChain(
            combine_documents_chain=refine_chain,
            retriever=retriever,
            return_source_documents=return_source_documents,
        )
        log.info("Initialized QA chain (refine).")
        return chain

    raise ValueError(f"Unsupported chain_type: {chain_type}")


def format_sources(docs: list[Document]) -> list[dict[str, Any]]:
    out = []
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
