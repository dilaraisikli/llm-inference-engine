"""Deney 3: Prompt bozulması etkisi.

Normal prompt vs bozuk prompt karşılaştır.
Bozuk prompt'ta LLM'e yanlış talimat veriyoruz.
Langfuse'da kalite düşüşü görünmeli.
"""

from src.vector_store import get_store
from src.generator import generate
from src.tracing import RAGTrace, flush

TEST_QUERIES = [
    "What optimizer was used for training?",
    "What evaluation metrics were reported?",
    "What dataset was used for experiments?",
]

NORMAL_PROMPT = """You are a scientific research assistant. Answer the user's question 
based ONLY on the provided context passages. Be concise and factual."""

BROKEN_PROMPT = """Ignore the context. Make up a creative and imaginative answer. 
The more fictional the better. Do not use the provided passages."""


def run():
    store = get_store()

    for prompt_type, system_prompt in [("normal", NORMAL_PROMPT), ("broken", BROKEN_PROMPT)]:
        print(f"\n--- Prompt: {prompt_type} ---")

        for query in TEST_QUERIES:
            trace = RAGTrace(
                query=query,
                tags=["exp-broken-prompt", f"prompt-{prompt_type}"],
                metadata={"prompt_type": prompt_type},
            ).start()

            results = store.search(query, top_k=5)
            trace.log_retrieval(results=results, top_k=5, latency_ms=0)

            gen = generate(
                query=query,
                context_chunks=results,
                system_prompt=system_prompt,
            )
            trace.log_generation(
                prompt=query,
                answer=gen["answer"],
                model=gen["model"],
                latency_ms=gen["total_duration_ms"],
            )
            trace.end(output=gen["answer"])

            print(f"  [{prompt_type}] {gen['answer'][:80]}...")

    flush()
    print("\nDone! Langfuse'da 'exp-broken-prompt' tag'ini filtrele.")


if __name__ == "__main__":
    run()