"""LLM generation module — AWS Bedrock Nova Micro.

Ollama yerine AWS Bedrock kullanıyoruz.
Model: amazon.nova-micro-v1:0
"""

import json
import boto3

from src.config import LLM_MODEL

# Bedrock client
_client = boto3.client("bedrock-runtime", region_name="eu-north-1")

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
    model: str = "amazon.nova-micro-v1:0",
    system_prompt: str = SYSTEM_PROMPT,
) -> dict:
    """AWS Bedrock ile cevap üret."""
    import time
    user_message = build_prompt(query, context_chunks)

    body = {
        "messages": [
            {"role": "user", "content": [{"text": user_message}]}
        ],
        "system": [{"text": system_prompt}],
        "inferenceConfig": {
            "temperature": 0.0,
            "maxTokens": 512,
        },
    }

    t0 = time.perf_counter()
    response = _client.invoke_model(
        modelId=model,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    duration_ms = (time.perf_counter() - t0) * 1000

    result = json.loads(response["body"].read())
    answer = result["output"]["message"]["content"][0]["text"]
    usage = result.get("usage", {})

    return {
        "answer": answer,
        "model": model,
        "total_duration_ms": duration_ms,
        "prompt_eval_count": usage.get("inputTokens", 0),
        "eval_count": usage.get("outputTokens", 0),
    }


def check_ollama_health() -> bool:
    """Bedrock için health check — her zaman True döner."""
    try:
        _client.list_foundation_models()
        return True
    except Exception:
        return False