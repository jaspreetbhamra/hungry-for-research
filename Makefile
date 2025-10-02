.PHONY: run test lint format index

run:
	uv run streamlit run src/app/main.py

index:
	uv run python -m ingestion.cli_build_index --input_dir data/docs --urls_file data/urls.txt --collection papers

test:
	uv run pytest tests/

lint:
	uv run ruff check .

format:
	uv run black src tests
	