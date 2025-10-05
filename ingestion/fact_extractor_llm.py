from __future__ import annotations

import re
from typing import Dict, Generator, Iterable, List

import instructor
import orjson
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from graph.neo4j_client import Neo4jClient
from ingestion.document_models import Chunk

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
- Output must be a JSON list of objects.
- Each object MUST have "subject", "predicate", "object".
- All fields must be non-empty strings.
- If multiple objects exist, output multiple triples.
- Do not return arrays inside "object" or "subject".
- Again, Every triple MUST include subject, predicate, object (all as strings). No missing fields.
- Example:
    [
    {{"subject": "Transformer", "predicate": "introduced_in", "object": "2017"}},
    {{"subject": "Transformer", "predicate": "applied_to", "object": "machine translation"}}
    ]

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
# Helper: unwrap JSON list
# --------------------
def unwrap_json_list(raw: str) -> str:
    """
    Extract the first valid JSON array [ ... ] from a raw LLM response.
    Returns a JSON string containing only that list, or "[]" if not found.
    """
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        return match.group(0)
    return "[]"


# --------------------
# Fact extraction
# --------------------
def extract_facts_from_chunks_llm(
    chunks: Iterable["Chunk"],
    client: OpenAI | None = None,
    model: str = "mistral",  # your Ollama model
    max_chunks: int | None = None,  # optional limiter for debugging
    show_progress: bool = True,
) -> Generator[Dict[str, str], None, None]:
    """
    Extract facts with one LLM call per chunk.
    Yields dicts incrementally: {"paper_id", "subject", "predicate", "object"}.
    """

    if client is None:
        client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            ),
            mode=instructor.Mode.JSON,
        )

    chunks_list = list(chunks)
    iterator = (
        tqdm(
            enumerate(chunks_list),
            total=len(chunks_list),
            desc="Extracting facts",
            unit="chunk",
        )
        if show_progress
        else enumerate(chunks_list)
    )

    for idx, chunk in iterator:
        if max_chunks and idx >= max_chunks:
            break

        # Each chunk gets its own prompt
        prompt = EXTRACTION_PROMPT.format(chunk=chunk.text[:1500])

        try:
            triples: List[Triple] = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_model=List[Triple],
            )

            for t in triples:
                yield {
                    "paper_id": chunk.metadata.get("source_id", "unknown"),
                    "subject": t.subject.strip(),
                    "predicate": t.predicate.strip(),
                    "object": t.object.strip(),
                }
        except Exception as e:
            print(f"[WARN] Failed to extract facts in chunk {idx}: {e}")


# --------------------
# Neo4j integration
# --------------------
def upsert_facts_to_neo4j(
    neo: "Neo4jClient",
    facts: Generator[Dict[str, str], None, None],
    show_progress: bool = False,
) -> None:
    """
    Stream triples into Neo4j as they are extracted.
    Facts is expected to be a generator from extract_facts_from_chunks_llm.
    """
    iterator = facts
    if show_progress:
        iterator = tqdm(facts, desc="Writing facts to Neo4j", unit="fact")

    for fact in iterator:
        try:
            neo.upsert_fact(
                paper_id=fact["paper_id"],
                subject=fact["subject"],
                predicate=fact["predicate"],
                obj=fact["object"],
            )
        except Exception as e:
            print(f"[WARN] Failed to insert fact {fact}: {e}")
