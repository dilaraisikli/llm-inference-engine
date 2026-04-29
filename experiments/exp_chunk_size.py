"""Deney 1: Chunk size etkisi.

128, 256, 512 kelimelik chunk'ları karşılaştır.
Her deney Langfuse'a farklı tag ile gider.
"""

from src.data_loader import load_from_disk
from src.chunker import chunk_corpus
from src.vector_store import FAISSStore
from src.generator import generate
from src.tracing import RAGTrace, flush

TEST_QUERIES = [
    "What is the seed lexicon?",           # QASPER'dan gerçek soru
    "How are relations used to propagate polarity?",
    "What are labels available in dataset for supervision?",
]

CHUNK_SIZES = [128, 256, 512]


def run():
    corpus = load_from_disk("qasper_corpus.jsonl")

    for chunk_size in CHUNK_SIZES:
        print(f"\n--- Chunk size: {chunk_size} ---")
        chunks = chunk_corpus(corpus, chunk_size=chunk_size, chunk_overlap=chunk_size // 8)
        store = FAISSStore.build_from_chunks(chunks)

        for query in TEST_QUERIES:
            trace = RAGTrace(
                query=query,
                tags=["exp-chunk-size", f"chunk-{chunk_size}"],
                metadata={"chunk_size": chunk_size},
            ).start()

            results = store.search(query, top_k=5)
            trace.log_retrieval(results=results, top_k=5, latency_ms=0)

            gen = generate(query=query, context_chunks=results)
            trace.log_generation(
                prompt=query,
                answer=gen["answer"],
                model=gen["model"],
                latency_ms=gen["total_duration_ms"],
            )
            trace.end(output=gen["answer"])

            print(f"  [{chunk_size}] top_score={results[0]['score']:.3f}")

    flush()
    print("\nDone! Langfuse'da 'exp-chunk-size' tag'ini filtrele.")


if __name__ == "__main__":
    run()