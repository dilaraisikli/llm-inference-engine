"""Deney 2: top_k etkisi.

top_k = 2, 5, 10 karşılaştır.
Daha fazla chunk daha iyi cevap verir mi?
"""

from src.vector_store import get_store
from src.generator import generate
from src.tracing import RAGTrace, flush

TEST_QUERIES = [
    "What is the seed lexicon?",         
    "How are relations used to propagate polarity?",
    "What are labels available in dataset for supervision?",
]

TOP_K_VALUES = [2, 5, 10]


def run():
    store = get_store()

    for top_k in TOP_K_VALUES:
        print(f"\n--- top_k: {top_k} ---")

        for query in TEST_QUERIES:
            trace = RAGTrace(
                query=query,
                tags=["exp-top-k", f"top-k-{top_k}"],
                metadata={"top_k": top_k},
            ).start()

            results = store.search(query, top_k=top_k)
            trace.log_retrieval(results=results, top_k=top_k, latency_ms=0)

            gen = generate(query=query, context_chunks=results)
            trace.log_generation(
                prompt=query,
                answer=gen["answer"],
                model=gen["model"],
                latency_ms=gen["total_duration_ms"],
            )
            trace.end(output=gen["answer"])

            print(f"  [top_k={top_k}] {query[:40]}...")

    flush()
    print("\nDone! Langfuse'da 'exp-top-k' tag'ini filtrele.")


if __name__ == "__main__":
    run()