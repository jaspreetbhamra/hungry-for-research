.PHONY: run test lint format

run:
	uv run streamlit run src/app/main.py

test:
	uv run pytest tests/

lint:
	uv run ruff check .

format:
	uv run black src tests
