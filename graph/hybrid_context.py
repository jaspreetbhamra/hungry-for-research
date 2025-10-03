from __future__ import annotations

from typing import Dict, List

from langchain.docstore.document import Document


def extract_candidate_entities(docs: List[Document], top_n: int = 5) -> List[str]:
    """
    Very naive entity extractor: pick top capitalized tokens from retrieved docs.
    Replace with spaCy or curated entity lists as needed.
    """
    import re
    from collections import Counter

    cap = re.compile(r"\b([A-Z][A-Za-z0-9\-]{2,})\b")
    ctr = Counter()
    for d in docs:
        for m in cap.findall(d.page_content):
            ctr[m] += 1
    return [w for w, _ in ctr.most_common(top_n)]


def format_graph_facts(rows: List[Dict]) -> str:
    lines = []
    for r in rows:
        subj = r.get("subject")
        pred = r.get("predicate")
        obj = r.get("object")
        pid = r.get("paper_id")
        src = f"(paper_id: {pid})" if pid else ""
        lines.append(f"- {subj} --{pred}--> {obj} {src}".strip())
    return "\n".join(lines)
