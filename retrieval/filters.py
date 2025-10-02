from __future__ import annotations

from typing import Dict, Iterable, Optional


def build_where_filter(
    sources: Optional[Iterable[str]] = None,
    types: Optional[Iterable[str]] = None,  # "pdf" | "text" | "web"
    min_page: Optional[int] = None,
    max_page: Optional[int] = None,
) -> Dict:
    """
    Construct a Chroma 'where' filter dict using metadata fields we wrote during ingestion:
      - metadata.source (filename or URL)
      - metadata.type ("pdf" | "text" | "web")
      - metadata.page (if pdf)
    """
    where: Dict = {}
    if sources:
        where["source"] = {"$in": list(sources)}
    if types:
        where["type"] = {"$in": list(types)}
    # page range uses an $and with comparison operators if provided
    page_filters = []
    if min_page is not None:
        page_filters.append({"page": {"$gte": min_page}})
    if max_page is not None:
        page_filters.append({"page": {"$lte": max_page}})
    if page_filters:
        # if we already had conditions, add them into $and
        if where:
            where = {"$and": [where, *page_filters]}
        else:
            where = {"$and": page_filters}
    return where
