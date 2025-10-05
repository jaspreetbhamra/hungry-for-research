from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Generator, Iterable, List

import orjson
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
- Each object MUST have "subject", "predicate", "object", "chunk_id" fields.
- Use the chunk_id marker provided in the text (e.g., [CHUNK_ID: doc123#chunk=7]).
- Do NOT invent chunk_ids; always copy them exactly from the input.
- All fields must be non-empty strings.
- If multiple objects exist, output multiple triples.
- Do not return arrays inside "object" or "subject".

Format:
[
  {{
    "subject": "<string>",
    "predicate": "<string>",
    "object": "<string>",
    "chunk_id": "<string>"
  }},
  ...
]

Examples:

Input:
[CHUNK_ID: doc123#chunk=0]
"This paper introduces the Transformer model, which uses the Adam optimizer."

Output:
[
  {{
    "subject": "Transformer",
    "predicate": "introduces",
    "object": "Transformer model",
    "chunk_id": "doc123#chunk=0"
  }},
  {{
    "subject": "Transformer",
    "predicate": "uses",
    "object": "Adam optimizer",
    "chunk_id": "doc123#chunk=0"
  }}
]

Input:
[CHUNK_ID: doc456#chunk=2]
"This work compares convolutional networks with recurrent networks for sequence modeling."

Output:
[
  {{
    "subject": "Convolutional networks",
    "predicate": "compares_with",
    "object": "Recurrent networks",
    "chunk_id": "doc456#chunk=2"
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
    chunk_id: str  # new field for traceability


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
    client=None,
    model: str = "mistral",
    batch_size: int = 5,
    max_batches: int | None = None,
    show_progress: bool = True,
) -> Generator[Dict[str, str], None, None]:
    """
    Option B (with chunk_id):
    - Groups chunks by document.
    - Tags chunks with CHUNK_ID.
    - LLM must return {"subject", "predicate", "object", "chunk_id"}.
    """

    if client is None:
        # lazy imports
        import instructor
        from openai import OpenAI

        client = instructor.from_openai(
            OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
            mode=instructor.Mode.JSON,
        )

    grouped: Dict[str, List[Chunk]] = defaultdict(list)
    for c in chunks:
        grouped[c.metadata["source_id"]].append(c)

    for doc_id, doc_chunks in grouped.items():
        batch_iter = list(batch_chunks(doc_chunks, size=batch_size))
        iterator = (
            tqdm(
                enumerate(batch_iter),
                total=len(batch_iter),
                desc=f"Extracting facts ({doc_id})",
                unit="batch",
            )
            if show_progress
            else enumerate(batch_iter)
        )

        for batch_idx, batch in iterator:
            if max_batches and batch_idx >= max_batches:
                break

            # tag chunks with [CHUNK_ID: ...]
            chunk_texts = []
            for c in batch:
                chunk_id = f"{doc_id}#chunk={c.metadata.get('chunk_index', 0)}"
                tagged = f"[CHUNK_ID: {chunk_id}]\n{c.text[:1500]}"
                chunk_texts.append(tagged)

            text = "\n\n---\n\n".join(chunk_texts)
            prompt = EXTRACTION_PROMPT.format(chunk=text)

            try:
                triples: List[Triple] = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=List[Triple],
                )

                for t in triples:
                    yield {
                        "paper_id": doc_id,
                        "chunk_id": t.chunk_id.strip(),
                        "subject": t.subject.strip(),
                        "predicate": t.predicate.strip(),
                        "object": t.object.strip(),
                    }
            except Exception as e:
                print(
                    f"[WARN] Failed to extract facts for doc={doc_id} batch={batch_idx}: {e}"
                )


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
                chunk_id=fact["chunk_id"],
            )
        except Exception as e:
            print(f"[WARN] Failed to insert fact {fact}: {e}")
