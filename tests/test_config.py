"""Unit tests for config.py"""

import os
import pytest


def test_config_imports():
    """Config modülü import edilebilmeli."""
    from src.config import (
        OLLAMA_BASE_URL,
        LLM_MODEL,
        EMBEDDING_MODEL,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        TOP_K,
    )
    assert OLLAMA_BASE_URL is not None
    assert LLM_MODEL is not None
    assert EMBEDDING_MODEL is not None


def test_config_defaults():
    """Varsayılan değerler doğru olmalı."""
    from src.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
    assert isinstance(CHUNK_SIZE, int)
    assert isinstance(CHUNK_OVERLAP, int)
    assert isinstance(TOP_K, int)
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP >= 0
    assert TOP_K > 0


def test_chunk_overlap_less_than_chunk_size():
    """Overlap chunk_size'dan küçük olmalı."""
    from src.config import CHUNK_SIZE, CHUNK_OVERLAP
    assert CHUNK_OVERLAP < CHUNK_SIZE


def test_ollama_url_format():
    """Ollama URL http ile başlamalı."""
    from src.config import OLLAMA_BASE_URL
    assert OLLAMA_BASE_URL.startswith("http")


def test_paths_exist():
    """BASE_DIR gerçek bir path olmalı."""
    from src.config import BASE_DIR
    assert BASE_DIR.exists()