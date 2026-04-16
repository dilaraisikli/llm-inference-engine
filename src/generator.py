"""LLM generation module — Ollama üzerinden Llama 3'e REST API ile konuşur.

Ollama local'de port 11434'te servis olarak çalışır.
Özel bir SDK yok — httpx ile doğrudan HTTP POST atarız.
"""

import httpx

from src.config import OLLAMA_BASE_URL, LLM_MODEL

# Reusable HTTP client — her çağrıda yeni client açmamak için
# timeout=120 çünkü CPU'da inference yavaş, cevap gelene kadar 1+ dakika sürebilir
_client = httpx.Client(base_url=OLLAMA_BASE_URL, timeout=120.0)


# --- Default system prompt ---
SYSTEM_PROMPT = """You are a scientific research assistant. Answer the user's question 
based ONLY on the provided context passages. If the context doesn't contain enough 
information to answer, say "I don't have enough information to answer this."

Rules:
- Be concise and factual.
- Cite which passage(s) support your answer.
- Do not make up facts."""


RAG_USER_TEMPLATE = """Context passages:
{context}

Question: {query}

Answer:"""


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """Retrieve edilmiş chunk'ları prompt'a yerleştir."""
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        context_text += f"\n[Passage {i}] {chunk['text']}\n"

    return RAG_USER_TEMPLATE.format(context=context_text, query=query)


def generate(
    query: str,
    context_chunks: list[dict],
    model: str = LLM_MODEL,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict:
    """Ollama ile cevap üret.

    Args:
        query: Kullanıcı sorusu.
        context_chunks: FAISS'ten gelen retrieve edilmiş chunk'lar.
        model: Ollama model adı (örn. "llama3.2:3b").
        system_prompt: Sistem yönergesi.

    Returns:
        dict: answer, model, total_duration_ms, prompt_eval_count, eval_count
    """
    user_message = build_prompt(query, context_chunks)

    response = _client.post(
        "/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,  # Streaming istemiyoruz — tek parça cevap
        },
    )
    response.raise_for_status()
    data = response.json()

    return {
        "answer": data.get("message", {}).get("content", ""),
        "model": data.get("model", model),
        "total_duration_ms": data.get("total_duration", 0) / 1_000_000,  # ns → ms
        "prompt_eval_count": data.get("prompt_eval_count", 0),  # input tokens
        "eval_count": data.get("eval_count", 0),  # output tokens
    }


def check_ollama_health() -> bool:
    """Ollama sunucusu ayakta mı kontrol et."""
    try:
        r = _client.get("/api/tags")
        return r.status_code == 200
    except httpx.ConnectError:
        return False


if __name__ == "__main__":
    # Ollama çalışıyor mu?
    if not check_ollama_health():
        print("⚠ Ollama erişilemiyor. 'ollama serve' çalışıyor mu?")
        exit(1)
    print("Ollama OK ✓")

    # FAISS'ten gerçek chunk'lar çek
    from src.vector_store import FAISSStore
    store = FAISSStore.load()

    # Test sorusu
    query = "What optimizer was used for training?"
    print(f"\nQuery: {query}")

    results = store.search(query, top_k=3)
    print(f"Retrieved {len(results)} chunks")

    # Llama 3'e sor
    print("\nGenerating answer with Llama 3...")
    result = generate(query=query, context_chunks=results)

    print("\n" + "="*60)
    print("ANSWER:")
    print(result["answer"])
    print("="*60)
    print(f"Model: {result['model']}")
    print(f"Duration: {result['total_duration_ms']:.0f} ms")
    print(f"Input tokens: {result['prompt_eval_count']}")
    print(f"Output tokens: {result['eval_count']}")