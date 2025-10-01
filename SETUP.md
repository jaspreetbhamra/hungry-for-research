# Local Model Setup

The system runs fully offline by combining a local FAISS index with embeddings and an
Ollama-hosted LLaMA model. Follow the steps below before launching the Streamlit app.

## 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

## 2. Cache the sentence-transformers embedding model
The retriever relies on `sentence-transformers/all-MiniLM-L6-v2`.
```bash
huggingface-cli login        # one-time auth, optional if the model is public for you
mkdir -p models/embeddings
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir models/embeddings/all-MiniLM-L6-v2 \
  --local-dir-use-symlinks False
```

## 3. Install and provision Ollama
1. Install Ollama following the official docs: https://ollama.com/download
   - macOS: `brew install --cask ollama`
   - Linux: run the provided install script from the site
2. Start the Ollama service (automatically when launching the desktop app, or manually via `ollama serve`).
3. Pull the desired LLaMA family model, e.g.:
```bash
ollama pull llama2
```

## 4. Smoke test the stack
Run a quick Python check to ensure both the embeddings and the Ollama-backed LLM load.
```python
from langchain_core.documents import Document
from src.vectorstore.faiss_store import build_faiss_store
from src.chains.qa_chain import load_ollama_llm

store = build_faiss_store([Document(page_content="Hello world")])
llm = load_ollama_llm(model="llama2")
print(llm.invoke("Say hello in one sentence."))
```

If the commands succeed, you are ready to process PDFs through the Streamlit interface.
