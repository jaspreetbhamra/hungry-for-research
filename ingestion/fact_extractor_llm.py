from __future__ import annotations

from typing import Dict, Generator, Iterable, List

from pydantic import BaseModel
from tqdm import tqdm

from graph.neo4j_client import Neo4jClient
from ingestion.document_models import Chunk
from models.llm import load_local_llm

# --- Prompt Template ---
EXTRACTION_PROMPT = """You are an expert information extraction system.

Task:
Extract only clear Subject–Predicate–Object facts about scientific methods, models, datasets, or algorithms.

Rules:
- Only include facts that express a technical relationship (e.g., "Transformer – uses – Adam Optimizer").
- Do NOT include vague descriptions ("Transformer is popular", "This paper is about...").
- If no clear facts exist, return an empty list.
- Use consistent predicates like ["uses", "introduces", "compares_with", "based_on"].
- Always output in strict JSON, no text before or after.

Format:
[
  {{
    "subject": "<string>",
    "predicate": "<string>",
    "object": "<string>"
  }},
  ...
]

Examples:

Input:
"This paper introduces the Transformer model, which uses the Adam optimizer."

Output:
[
  {{
    "subject": "Transformer",
    "predicate": "introduces",
    "object": "Transformer model"
  }},
  {{
    "subject": "Transformer",
    "predicate": "uses",
    "object": "Adam optimizer"
  }}
]

Now process the following text:

{chunk}

"""


# --------------------
# Validation schema
# --------------------
class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


def validate_triples(raw_json: str) -> List[Triple]:
    import orjson

    try:
        data = orjson.loads(raw_json)
        if not isinstance(data, list):
            return []
        return [Triple(**t) for t in data if isinstance(t, dict)]
    except Exception as e:
        print(f"[WARN] Failed to parse triples: {e}")
        return []


# --------------------
# Batch utilities
# --------------------
def batch_chunks(chunks: List[Chunk], size: int = 5):
    """Yield batches of chunks for more efficient LLM calls."""
    for i in range(0, len(chunks), size):
        yield chunks[i : i + size]


# --------------------
# Fact extraction
# --------------------
def extract_facts_from_chunks_llm(
    chunks: Iterable[Chunk],
    llm=None,
    batch_size: int = 5,
    max_batches: int | None = None,
    show_progress: bool = True,
) -> Generator[Dict[str, str], None, None]:
    """
    Use an LLM to extract structured triples from text chunks.
    Yields dicts incrementally: {"paper_id", "subject", "predicate", "object"}.
    """
    if llm is None:
        llm = load_local_llm("llm_extraction")

    batch_iter = list(batch_chunks(list(chunks), size=batch_size))
    iterator = (
        tqdm(
            enumerate(batch_iter),
            total=len(batch_iter),
            desc="Extracting facts",
            unit="batch",
        )
        if show_progress
        else enumerate(batch_iter)
    )

    for batch_idx, batch in iterator:
        if max_batches and batch_idx >= max_batches:
            break

        text = "\n\n---\n\n".join(c.text[:1500] for c in batch)
        prompt = EXTRACTION_PROMPT.format(chunk=text)

        try:
            response = llm.invoke(prompt)
            valid_triples = validate_triples(response.strip())
            for t in valid_triples:
                yield {
                    "paper_id": batch[0].metadata.get("source_id", "unknown"),
                    "subject": t.subject.strip(),
                    "predicate": t.predicate.strip(),
                    "object": t.object.strip(),
                }
        except Exception as e:
            print(f"[WARN] Failed to extract facts in batch {batch_idx}: {e}")


# --------------------
# Neo4j integration
# --------------------
def upsert_facts_to_neo4j(neo: Neo4jClient, facts: List[Dict[str, str]]) -> None:
    """
    Push extracted triples into Neo4j.
    """
    for fact in facts:
        try:
            neo.upsert_fact(
                paper_id=fact["paper_id"],
                subject=fact["subject"],
                predicate=fact["predicate"],
                obj=fact["object"],
            )
        except Exception as e:
            print(f"[WARN] Failed to insert fact {fact}: {e}")
