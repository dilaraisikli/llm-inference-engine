"""Unit tests for API Pydantic schemas."""

import pytest
from pydantic import ValidationError
from api.schemas import AskRequest, AskResponse, SourceChunk, HealthResponse


def test_ask_request_valid():
    """Geçerli request kabul edilmeli."""
    req = AskRequest(query="What optimizer was used?", top_k=5)
    assert req.query == "What optimizer was used?"
    assert req.top_k == 5
    assert req.tags == []


def test_ask_request_default_top_k():
    """top_k varsayılan 5 olmalı."""
    req = AskRequest(query="test question")
    assert req.top_k == 5


def test_ask_request_empty_query_fails():
    """Boş query reddedilmeli."""
    with pytest.raises(ValidationError):
        AskRequest(query="")


def test_ask_request_top_k_min():
    """top_k minimum 1 olmalı."""
    with pytest.raises(ValidationError):
        AskRequest(query="test", top_k=0)


def test_ask_request_top_k_max():
    """top_k maksimum 20 olmalı."""
    with pytest.raises(ValidationError):
        AskRequest(query="test", top_k=21)


def test_ask_request_with_tags():
    """Tags listesi doğru şekilde alınmalı."""
    req = AskRequest(query="test", tags=["exp-chunk-size", "chunk-256"])
    assert req.tags == ["exp-chunk-size", "chunk-256"]


def test_source_chunk_valid():
    """Geçerli SourceChunk oluşturulabilmeli."""
    chunk = SourceChunk(
        chunk_id="chunk_00001",
        doc_id="paper_123",
        title="Test Paper",
        score=0.85,
        text="Some text here",
    )
    assert chunk.score == 0.85


def test_ask_response_valid():
    """Geçerli AskResponse oluşturulabilmeli."""
    response = AskResponse(
        answer="The optimizer was Adam.",
        sources=[],
        retrieval_ms=123.4,
        generation_ms=5678.9,
        model="llama3.1:8b",
    )
    assert response.answer == "The optimizer was Adam."


def test_health_response_healthy():
    """Healthy status doğru oluşturulmalı."""
    health = HealthResponse(
        status="healthy",
        ollama=True,
        index_loaded=True,
        index_size=3421,
    )
    assert health.status == "healthy"
    assert health.index_size == 3421


def test_health_response_degraded():
    """Degraded status doğru oluşturulmalı."""
    health = HealthResponse(
        status="degraded",
        ollama=False,
        index_loaded=True,
        index_size=3421,
    )
    assert health.status == "degraded"
    assert health.ollama == False