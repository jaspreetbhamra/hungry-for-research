.PHONY: dev clean run run_app index query hybrid_query hybrid_answer test_extract test lint format neo_clear

# -----------------
# Setup / Cleanup
# -----------------
dev:
	uv venv .venv --python=3.12
	uv pip install -e ".[dev]"

clean:
	rm -rf .venv build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +

# -----------------
# Streamlit apps
# -----------------
run:
	uv run streamlit run app/streamlit_app.py

run_app: run  # alias for consistency

# -----------------
# Pipelines
# -----------------
index:
	export TOKENIZERS_PARALLELISM=false; \
	uv run python -m ingestion.cli_build_index \
		--input_dir data/docs \
		--urls_file data/urls.txt \
		--collection papers

query:
	uv run python -m chains.cli_query \
		--collection papers \
		--mode mmr --k 5 --fetch_k 30 --lambda_mult 0.4 \
		--chain_type stuff \
		"What does ResNet change versus VGG?"

hybrid_query:
	uv run python -m chains.cli_hybrid_query \
		--neo "What optimizers are mentioned in Transformer papers?"

hybrid_answer:
	uv run python -m chains.cli_hybrid_answer \
		--mode mmr --k 5 --fetch_k 30 --lambda_mult 0.4 \
		"How does ResNet differ from VGG?"

# -----------------
# Neo4j Utilities
# -----------------
neo_clear:
	uv run python -c "from graph.neo4j_client import Neo4jClient; \
	c=Neo4jClient(); \
	c.run('MATCH (n) DETACH DELETE n'); \
	c.close(); \
	print('Neo4j database cleared.')"

# -----------------
# Testing
# -----------------
test_extract:
	uv run python -m scripts.test_extraction

test:
	uv run pytest tests/

# -----------------
# Quality
# -----------------
lint:
	uv run ruff check .

format:
	uv run black app common ingestion graph models retrieval vectorstore tests
