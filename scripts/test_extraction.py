from ingestion.chunkers import chunk_documents
from ingestion.document_models import RawDoc
from ingestion.fact_extractor_llm import extract_facts_from_chunks_llm


def load_sample_text():
    """Return a dummy RawDoc for testing."""
    return RawDoc(
        source_id="test_doc",
        text="""This paper introduces the Transformer model,
                which uses the Adam optimizer for training.
                Later, BERT was based on the Transformer architecture.""",
        metadata={"source": "test.txt", "type": "text"},
        content_sha1="dummyhash",
    )


def main():
    doc = load_sample_text()
    chunks = chunk_documents([doc])

    print("Running fact extraction...")
    for fact in extract_facts_from_chunks_llm(chunks, batch_size=1, max_batches=1):
        print(fact)


if __name__ == "__main__":
    main()
