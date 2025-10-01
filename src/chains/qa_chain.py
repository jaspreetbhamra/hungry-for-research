"""Question-answering chains built on top of FAISS retrievers."""
from __future__ import annotations

from typing import Any

from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseLanguageModel


def load_ollama_llm(
    model: str = "llama2",
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    request_timeout: int | None = 120,
    **kwargs: Any,
) -> BaseLanguageModel:
    """Return a LangChain LLM wired to a locally running Ollama instance.

    Args:
        model: Name of the Ollama model to run (e.g. ``"llama2"``).
        temperature: Sampling temperature for generation.
        max_tokens: Maximum number of tokens to generate per response.
        request_timeout: Optional client timeout passed to the Ollama LLM.
        **kwargs: Additional keyword arguments forwarded to ``Ollama``.

    Returns:
        A ``BaseLanguageModel`` implementation backed by Ollama.

    Raises:
        ConnectionError: If the Ollama server cannot be reached.
    """

    return Ollama(
        model=model,
        temperature=temperature,
        num_predict=max_tokens,
        request_timeout=request_timeout,
        **kwargs,
    )


def build_qa_chain(
    llm: BaseLanguageModel,
    vectorstore: FAISS,
    *,
    k: int = 4,
    chain_type: str = "stuff",
    chain_kwargs: dict[str, Any] | None = None,
    return_source_documents: bool = False,
) -> RetrievalQA:
    """Link an offline LLM with a FAISS-backed retriever for question answering.

    Args:
        llm: Language model to generate answers with.
        vectorstore: FAISS vector store providing similarity search.
        k: Number of top documents to retrieve per question.
        chain_type: LangChain chain type (e.g. ``"stuff"`` or ``"map_reduce"``).
        chain_kwargs: Optional kwargs forwarded to ``RetrievalQA.from_chain_type``.
        return_source_documents: Whether to include source documents in results.

    Returns:
        A configured ``RetrievalQA`` chain ready for invocation.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        chain_type_kwargs=chain_kwargs or {},
        return_source_documents=return_source_documents,
    )
