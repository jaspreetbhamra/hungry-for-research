from langchain.prompts import PromptTemplate

SYSTEM_BASE = """You are a precise research assistant. Use ONLY the provided context to answer.
If the answer is not fully contained in the context, say you don't know.
Cite sources using (source: <source>, page: <page>) for each supporting snippet."""

STUFF_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        SYSTEM_BASE + "\n\n"
        "Question:\n{question}\n\n"
        "Context:\n{context}\n\n"
        "Answer with citations:"
    ),
)

MAP_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        SYSTEM_BASE + "\n\n"
        "You will be given a chunk of context. Extract anything relevant to answer:\n"
        "Question: {question}\n\n"
        "Context chunk:\n{context}\n\n"
        "Partial answer (with citations):"
    ),
)

REDUCE_PROMPT = PromptTemplate(
    input_variables=["question", "summaries"],
    template=(
        SYSTEM_BASE + "\n\n"
        "Combine the following partial answers into a single, concise final answer.\n"
        "Question: {question}\n\n"
        "Partial answers:\n{summaries}\n\n"
        "Final answer (with citations):"
    ),
)

REFINE_PROMPT_QUESTION = PromptTemplate(
    input_variables=["question", "existing_answer"],
    template=(
        SYSTEM_BASE + "\n\n"
        "We have an existing draft answer. Improve it using the new context if helpful.\n"
        "Question: {question}\n\n"
        "Existing answer:\n{existing_answer}\n"
        "New context:\n{{context}}\n\n"
        "Refined answer (with citations):"
    ),
)
