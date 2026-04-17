"""RAG Pipeline orchestrator.

Tüm parçaları birleştiren ana modül:
    Soru gelir → FAISS'ten chunk ara → Llama 3'e ver → cevap al → Langfuse'a trace gönder
"""

import time

from src.config import TOP_K
from src.vector_store import get_store
from src.generator import generate, build_prompt
from src.tracing import RAGTrace, flush


def ask(
    query: str,
    top_k: int = TOP_K,
    tags: list[str] | None = None,
    system_prompt: str | None = None,
) -> dict:
    """RAG pipeline'ın tek giriş noktası.

    Args:
        query: Kullanıcı sorusu.
        top_k: Kaç chunk retrieve edilsin.
        tags: Langfuse'da bu trace'i etiketlemek için (örn. ["exp-chunk-size"]).
        system_prompt: Varsayılan prompt yerine başka bir prompt kullan (deneyler için).

    Returns:
        dict: answer, sources, retrieval_ms, generation_ms, model
    """
    # --- Trace başlat ---
    trace = RAGTrace(query=query, tags=tags or ["api"]).start()

    # --- 1. Retrieval: FAISS'ten en yakın chunk'ları bul ---
    t0 = time.perf_counter()
    store = get_store()
    results = store.search(query, top_k=top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    trace.log_retrieval(results=results, top_k=top_k, latency_ms=retrieval_ms)

    # --- 2. Generation: Llama 3 ile cevap üret ---
    t1 = time.perf_counter()
    gen_kwargs = {"query": query, "context_chunks": results}
    if system_prompt:
        gen_kwargs["system_prompt"] = system_prompt
    gen_result = generate(**gen_kwargs)
    generation_ms = (time.perf_counter() - t1) * 1000

    prompt_text = build_prompt(query, results)
    trace.log_generation(
        prompt=prompt_text,
        answer=gen_result["answer"],
        model=gen_result["model"],
        latency_ms=generation_ms,
        prompt_tokens=gen_result.get("prompt_eval_count", 0),
        completion_tokens=gen_result.get("eval_count", 0),
    )

    # --- 3. Trace'i kapat ---
    trace.end(output=gen_result["answer"])

    return {
        "answer": gen_result["answer"],
        "sources": [
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "title": r["title"],
                "score": r["score"],
                "text": r["text"][:200],
            }
            for r in results
        ],
        "retrieval_ms": round(retrieval_ms, 1),
        "generation_ms": round(generation_ms, 1),
        "model": gen_result["model"],
    }


if __name__ == "__main__":
    query = "What evaluation metrics were used in the experiments?"
    print(f"Query: {query}\n")

    result = ask(query)

    print("="*60)
    print(f"ANSWER: {result['answer']}")
    print("="*60)
    print(f"Retrieval: {result['retrieval_ms']} ms")
    print(f"Generation: {result['generation_ms']} ms")
    print(f"Model: {result['model']}")
    print(f"\nSources ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  [{s['score']:.3f}] {s['title'][:50]} → {s['text'][:80]}...")

    flush()