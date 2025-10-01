ENV_NAME=hungry_for_research

.PHONY: setup activate freeze clean help

# Create or update the conda environment
setup:
	conda env create -f environment.yml || conda env update -f environment.yml --prune

# Activate the conda environment (prints instructions)
activate:
	@echo "Run the following command to activate the environment:"
	@echo "conda activate $(ENV_NAME)"

# Export a clean environment.yml from current env
freeze:
	conda env export --from-history > environment.yml
	@echo "Updated environment.yml from history (clean, no transitive deps)."

# Remove the conda environment
clean:
	conda deactivate
	conda env remove -n $(ENV_NAME)

# Show available commands
help:
	@echo "Available commands:"
	@echo "  make setup    - Create or update the environment"
	@echo "  make activate - Print activation instructions"
	@echo "  make freeze   - Export clean environment.yml"
	@echo "  make clean    - Remove the environment"
