"""FAISS vector store — build, save, load, and search.

Akış: chunks → embed → FAISS index + metadata → diske kaydet
Arama: query → embed → FAISS'te en yakın k vektörü bul → metadata'dan chunk bilgisini getir
"""

import json
from pathlib import Path

import faiss
import numpy as np

from src.config import INDEX_DIR, TOP_K
from src.embedder import embed_texts, embed_query


class FAISSStore:
    """FAISS index + metadata wrapper."""

    def __init__(self, index: faiss.Index, metadata: list[dict]):
        self.index = index
        self.metadata = metadata  # metadata[i] = i. vektörün chunk bilgisi

    @classmethod
    def build_from_chunks(cls, chunks: list[dict]) -> "FAISSStore":
        """Chunk'ları embed et ve FAISS index'i oluştur.

        Args:
            chunks: list of dicts with keys chunk_id, doc_id, title, text.
        """
        texts = [c["text"] for c in chunks] 
        print(f"Embedding {len(texts)} chunks...")
        embeddings = embed_texts(texts)

        dim = embeddings.shape[1]  # 384
        # IndexFlatIP = Inner Product (dot product)
        # Vektörler normalize olduğu için dot product = cosine similarity
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

        print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
        return cls(index=index, metadata=chunks)

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Sorguyu embed et, en yakın top_k chunk'ı getir.

        Returns:
            List of dicts: chunk bilgisi + similarity score.
        """
        query_vec = embed_query(query).astype(np.float32)
        scores, indices = self.index.search(query_vec, top_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS bulamadıysa -1 döner
                continue
            result = {**self.metadata[idx], "score": float(score)}
            results.append(result)

        return results

    def save(self, path: Path | None = None) -> None:
        """Index ve metadata'yı diske kaydet."""
        save_dir = path or INDEX_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(save_dir / "index.faiss"))

        with open(save_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        print(f"Index saved to {save_dir}")

    @classmethod
    def load(cls, path: Path | None = None) -> "FAISSStore":
        """Diskten index ve metadata'yı yükle."""
        load_dir = path or INDEX_DIR

        index = faiss.read_index(str(load_dir / "index.faiss"))

        metadata: list[dict] = []
        with open(load_dir / "metadata.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                metadata.append(json.loads(line.strip()))

        print(f"Index loaded: {index.ntotal} vectors")
        return cls(index=index, metadata=metadata)


# Module-level singleton — API'de tekrar tekrar yüklememek için
_store: FAISSStore | None = None


def get_store() -> FAISSStore:
    """Global FAISS store'u getir veya yükle."""
    global _store
    if _store is None:
        _store = FAISSStore.load()
    return _store


if __name__ == "__main__":
    from src.data_loader import load_from_disk

    # 1. Chunk'ları yükle
    chunks = load_from_disk("qasper_chunks.jsonl")
    print(f"Loaded {len(chunks)} chunks")

    # 2. Embed et + FAISS index oluştur
    store = FAISSStore.build_from_chunks(chunks)

    # 3. Diske kaydet
    store.save()

    # 4. Sanity check — bir soru sor
    print("\n--- Sanity check ---")
    results = store.search("What optimizer was used for training?")
    for r in results[:3]:
        print(f"  [{r['score']:.3f}] {r['text'][:120]}...")