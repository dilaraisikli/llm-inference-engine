"""Deney 4: Cache etkisi.

Aynı soruyu cache'li ve cache'siz çalıştır.
Cache hit'te generation atlanır — çok daha hızlı.
Langfuse'da generation_ms farkı görünmeli.
"""

import time
from src.vector_store import get_store
from src.generator import generate
from src.tracing import RAGTrace, flush

# Basit in-memory cache: query → answer
_cache: dict[str, str] = {}

TEST_QUERIES = [
    "What is the seed lexicon?",     
    "What is the seed lexicon?",    
    "How are relations used to propagate polarity?",
    "How are relations used to propagate polarity?"
]


def ask_with_cache(query: str, store, tags: list[str]) -> dict:
    """Cache'li RAG pipeline."""
    trace = RAGTrace(query=query, tags=tags, metadata={"cached": query in _cache}).start()

    t0 = time.perf_counter()
    results = store.search(query, top_k=5)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    trace.log_retrieval(results=results, top_k=5, latency_ms=retrieval_ms)

    if query in _cache:
        # Cache hit — LLM çağrısı yok
        answer = _cache[query]
        generation_ms = 0.0
        print(f"  CACHE HIT: {query[:40]}...")
    else:
        # Cache miss — LLM çağır
        t1 = time.perf_counter()
        gen = generate(query=query, context_chunks=results)
        generation_ms = (time.perf_counter() - t1) * 1000
        answer = gen["answer"]
        _cache[query] = answer
        print(f"  CACHE MISS: {query[:40]}...")

    trace.log_generation(
        prompt=query,
        answer=answer,
        model="llama3.2:3b",
        latency_ms=generation_ms,
    )
    trace.end(output=answer)

    return {
        "answer": answer,
        "retrieval_ms": round(retrieval_ms, 1),
        "generation_ms": round(generation_ms, 1),
        "cache_hit": query in _cache,
    }


def run():
    store = get_store()

    print("=== Cache Experiment ===")
    for query in TEST_QUERIES:
        result = ask_with_cache(query, store, tags=["exp-cache"])
        print(f"  retrieval={result['retrieval_ms']}ms, generation={result['generation_ms']}ms")

    flush()
    print("\nDone! Langfuse'da 'exp-cache' tag'ini filtrele.")
    print("Cache hit'lerde generation_ms=0 görünmeli.")


if __name__ == "__main__":
    run()