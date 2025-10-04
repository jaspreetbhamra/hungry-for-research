.PHONY: run test lint format index

run:
	uv run streamlit run src/app/main.py

index:
	uv run python -m ingestion.cli_build_index --input_dir data/docs --urls_file data/urls.txt --collection papers

query:
	uv run python -m chains.cli_query --collection papers --mode mmr --k 5 --fetch_k 30 --lambda_mult 0.4 --chain_type stuff "What does ResNet change versus VGG?"

hybrid_query:
	uv run python -m chains.cli_hybrid_query --neo "What optimizers are mentioned in Transformer papers?"

test_extract:
	uv run python -m scripts.test_extraction

test:
	uv run pytest tests/

lint:
	uv run ruff check .

format:
	uv run black src tests
	