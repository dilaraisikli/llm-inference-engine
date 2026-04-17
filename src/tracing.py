"""Langfuse tracing — Langfuse 4.x (OpenTelemetry tabanlı).

Langfuse 4.x'te get_client() ve start_as_current_observation kullanılır.
"""

import os
from src.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

_configured = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

if _configured:
    os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST


def get_langfuse():
    if not _configured:
        return None
    try:
        from langfuse import get_client
        return get_client()
    except Exception:
        return None


class RAGTrace:
    """Tek bir RAG isteğinin trace'ini yönetir."""

    def __init__(
        self,
        query: str,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ):
        self.query = query
        self.tags = tags or []
        self.metadata = metadata or {}
        self._lf = get_langfuse()
        self._root_span = None
        self._ctx = None

    def start(self) -> "RAGTrace":
        if self._lf is None:
            return self
        try:
            self._ctx = self._lf.start_as_current_observation(
                as_type="span",
                name="rag-pipeline",
                input=self.query,
                metadata={**self.metadata, "tags": self.tags},
            )
            self._root_span = self._ctx.__enter__()
        except Exception as e:
            print(f"Langfuse start error (non-fatal): {e}")
        return self

    def log_retrieval(
        self,
        results: list[dict],
        top_k: int,
        latency_ms: float,
    ) -> None:
        if self._lf is None:
            return
        try:
            with self._lf.start_as_current_observation(
                as_type="span",
                name="retrieval",
                input={"query": self.query, "top_k": top_k},
            ) as span:
                span.update(
                    output={
                        "num_results": len(results),
                        "top_scores": [r.get("score", 0) for r in results[:3]],
                        "chunk_ids": [r.get("chunk_id", "") for r in results],
                    },
                    metadata={"latency_ms": latency_ms},
                )
        except Exception as e:
            print(f"Langfuse retrieval log error (non-fatal): {e}")

    def log_generation(
        self,
        prompt: str,
        answer: str,
        model: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        if self._lf is None:
            return
        try:
            with self._lf.start_as_current_observation(
                as_type="generation",
                name="llm-generation",
                model=model,
                input=prompt,
            ) as gen:
                gen.update(
                    output=answer,
                    usage={"input": prompt_tokens, "output": completion_tokens},
                    metadata={"latency_ms": latency_ms},
                )
        except Exception as e:
            print(f"Langfuse generation log error (non-fatal): {e}")

    def end(self, output: str) -> None:
        if self._root_span is None or self._ctx is None:
            return
        try:
            self._root_span.update(output=output)
            self._ctx.__exit__(None, None, None)
        except Exception as e:
            print(f"Langfuse end error (non-fatal): {e}")

    def score(self, name: str, value: float, comment: str = "") -> None:
        if self._lf is None:
            return
        try:
            self._lf.score(name=name, value=value, comment=comment)
        except Exception as e:
            print(f"Langfuse score error (non-fatal): {e}")


def flush() -> None:
    lf = get_langfuse()
    if lf:
        try:
            lf.flush()
        except Exception as e:
            print(f"Langfuse flush error (non-fatal): {e}")