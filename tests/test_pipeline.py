from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def test_end_to_end_pipeline(tmp_path):
    # 1. Load sample text
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text(
        "Machine learning is great. Deep learning is a subset of machine learning."
    )

    loader = TextLoader(str(sample_file))
    documents = loader.load()

    # 2. Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    docs = splitter.split_documents(documents)

    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Chroma vector store
    db = Chroma.from_documents(docs, embeddings, persist_directory=str(tmp_path))

    # 5. Simple retrieval
    retriever = db.as_retriever(search_kwargs={"k": 1})
    results = retriever.get_relevant_documents("What is deep learning?")

    assert len(results) > 0
    assert "deep learning" in results[0].page_content.lower()
