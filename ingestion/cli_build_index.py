from __future__ import annotations

import argparse
from pathlib import Path

from common.logger import get_logger
from ingestion.ingest_pipeline import ingest_folder

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build/Update Chroma index from local docs and URLs list."
    )
    parser.add_argument(
        "--input_dir", type=str, default="data/docs", help="Folder with PDFs/TXT/MD"
    )
    parser.add_argument(
        "--urls_file",
        type=str,
        default="",
        help="Optional file with URLs (one per line)",
    )
    parser.add_argument(
        "--collection", type=str, default="papers", help="Chroma collection name"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    urls_file = Path(args.urls_file) if args.urls_file else None

    if not input_dir.exists():
        log.error("Input directory does not exist: %s", input_dir)
        raise SystemExit(1)

    ingest_folder(
        input_dir=input_dir, urls_file=urls_file, collection_name=args.collection
    )


if __name__ == "__main__":
    main()
