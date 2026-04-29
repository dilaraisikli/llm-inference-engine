"""Unit tests for data_loader save/load functions."""

import json
import pytest
from pathlib import Path
from src.data_loader import save_to_disk, load_from_disk


@pytest.fixture
def tmp_data(tmp_path):
    """Test için geçici data."""
    return [
        {"doc_id": "001", "title": "Paper 1", "text": "Some text"},
        {"doc_id": "002", "title": "Paper 2", "text": "Other text"},
    ]


def test_save_to_disk_creates_file(tmp_path, tmp_data):
    """save_to_disk dosya oluşturmalı."""
    out_path = tmp_path / "test.jsonl"
    save_to_disk(tmp_data, "test.jsonl", path=out_path)
    assert out_path.exists()


def test_save_to_disk_correct_count(tmp_path, tmp_data):
    """Kaydedilen satır sayısı data sayısıyla eşleşmeli."""
    out_path = tmp_path / "test.jsonl"
    save_to_disk(tmp_data, "test.jsonl", path=out_path)
    lines = out_path.read_text().strip().split("\n")
    assert len(lines) == len(tmp_data)


def test_load_from_disk_returns_same_data(tmp_path, tmp_data):
    """Kaydedip yükleme aynı veriyi döndürmeli."""
    out_path = tmp_path / "test.jsonl"
    save_to_disk(tmp_data, "test.jsonl", path=out_path)
    loaded = load_from_disk("test.jsonl", path=out_path)
    assert loaded == tmp_data


def test_save_load_preserves_all_fields(tmp_path):
    """Tüm alanlar korunmalı."""
    data = [{"doc_id": "001", "title": "T", "text": "Hello", "extra": 42}]
    out_path = tmp_path / "test.jsonl"
    save_to_disk(data, "test.jsonl", path=out_path)
    loaded = load_from_disk("test.jsonl", path=out_path)
    assert loaded[0]["extra"] == 42


def test_save_empty_list(tmp_path):
    """Boş liste kaydedilip yüklenebilmeli."""
    out_path = tmp_path / "empty.jsonl"
    save_to_disk([], "empty.jsonl", path=out_path)
    loaded = load_from_disk("empty.jsonl", path=out_path)
    assert loaded == []