from __future__ import annotations

import json
from typing import Dict, Generator, Iterable, List

import tqdm
from pydantic import BaseModel, ValidationError

from graph.neo4j_client import Neo4jClient
from ingestion.document_models import Chunk
from models.llm import load_local_llm

# --- Prompt Template ---
EXTRACTION_PROMPT = """You are an information extraction system.
Extract key factual relationships from the text below as (subject, predicate, object) triples.
Only include concise scientific or technical relations (e.g., "ResNet --uses--> BatchNorm").

Text:
{chunk}

Output ONLY a JSON list of triples, each formatted as:
[
  {{"subject": "...", "predicate": "...", "object": "..."}}
]
"""


# --------------------
# Validation schema
# --------------------
class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


def validate_triples(raw: str) -> List[Triple]:
    """
    Validate LLM JSON output against Triple schema.
    Returns a list of valid Triple objects.
    """
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        triples = [Triple(**t) for t in data if isinstance(t, dict)]
        return triples
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"[WARN] Skipping invalid output: {e}")
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
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
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
