"""Pydantic request/response modelleri."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    tags: list[str] = Field(default_factory=list)


class SourceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    retrieval_ms: float
    generation_ms: float
    model: str


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    index_loaded: bool
    index_size: int