"""Unit tests for chunker.py"""

import pytest
from src.chunker import chunk_by_tokens, chunk_by_sentences, chunk_corpus


def test_chunk_by_tokens_short_text():
    """Kısa text chunk'lanmamalı — zaten chunk_size altında."""
    text = "This is a short text."
    result = chunk_by_tokens(text, chunk_size=256, chunk_overlap=32)
    assert len(result) == 1
    assert result[0] == text


def test_chunk_by_tokens_long_text():
    """Uzun text birden fazla chunk'a bölünmeli."""
    words = ["word"] * 300
    text = " ".join(words)
    result = chunk_by_tokens(text, chunk_size=100, chunk_overlap=10)
    assert len(result) > 1


def test_chunk_by_tokens_overlap():
    """Overlap sayesinde chunk'lar arasında ortak kelimeler olmalı."""
    words = [f"word{i}" for i in range(200)]
    text = " ".join(words)
    chunks = chunk_by_tokens(text, chunk_size=100, chunk_overlap=20)

    # İlk chunk'ın son 20 kelimesi ikinci chunk'ın başında olmalı
    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    overlap_words = first_chunk_words[-20:]
    assert overlap_words == second_chunk_words[:20]


def test_chunk_by_tokens_no_empty_chunks():
    """Hiçbir chunk boş olmamalı."""
    text = "word " * 500
    chunks = chunk_by_tokens(text, chunk_size=100, chunk_overlap=10)
    for chunk in chunks:
        assert chunk.strip() != ""


def test_chunk_by_sentences_short_text():
    """Kısa text (az cümle) chunk'lanmamalı."""
    text = "First sentence. Second sentence."
    result = chunk_by_sentences(text, max_sentences=5, overlap_sentences=1)
    assert len(result) == 1


def test_chunk_by_sentences_splits_correctly():
    """Cümle bazlı chunk'lama cümle sınırlarına saygı duymalı."""
    sentences = [f"This is sentence {i}." for i in range(20)]
    text = " ".join(sentences)
    chunks = chunk_by_sentences(text, max_sentences=5, overlap_sentences=1)
    assert len(chunks) > 1


def test_chunk_corpus_preserves_doc_id():
    """chunk_corpus doc_id'yi korumalı."""
    corpus = [
        {"doc_id": "paper_001", "title": "Test Paper", "text": "word " * 300},
        {"doc_id": "paper_002", "title": "Another Paper", "text": "word " * 300},
    ]
    chunks = chunk_corpus(corpus, chunk_size=100, chunk_overlap=10)

    doc_ids = {c["doc_id"] for c in chunks}
    assert "paper_001" in doc_ids
    assert "paper_002" in doc_ids


def test_chunk_corpus_unique_chunk_ids():
    """Her chunk'ın benzersiz chunk_id'si olmalı."""
    corpus = [
        {"doc_id": "paper_001", "title": "Test", "text": "word " * 300},
    ]
    chunks = chunk_corpus(corpus, chunk_size=100, chunk_overlap=10)
    chunk_ids = [c["chunk_id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_corpus_empty_corpus():
    """Boş corpus boş liste döndürmeli."""
    result = chunk_corpus([], chunk_size=256, chunk_overlap=32)
    assert result == []


def test_chunk_corpus_has_required_keys():
    """Her chunk chunk_id, doc_id, title, text içermeli."""
    corpus = [{"doc_id": "p1", "title": "T", "text": "word " * 50}]
    chunks = chunk_corpus(corpus, chunk_size=100, chunk_overlap=10)
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "doc_id" in chunk
        assert "title" in chunk
        assert "text" in chunk