"""Embedding module using sentence-transformers.

Text → 384 boyutlu vektör.
Model: all-MiniLM-L6-v2 (80MB, CPU-friendly).
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

# Module-level singleton — model bir kere yüklenir, tekrar tekrar kullanılır
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of texts into dense vectors.

    Args:
        texts: List of chunk texts.
        batch_size: Kaç chunk'ı aynı anda embed eder.

    Returns:
        numpy array, shape: (len(texts), 384)
    """
    model = get_model()
    embeddings = model.encode(  #Modelin encode metodu — text'leri vektöre çeviren asıl fonksiyon
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Normalize → dot product = cosine similarity
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string.

    Returns:
        numpy array, shape: (1, 384)
    """
    model = get_model()
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding


if __name__ == "__main__":
    # Quick test
    test_texts = [
        "Adam optimizer with learning rate 3e-5",
        "SGD with momentum 0.9",
        "BLEU score for machine translation evaluation",
    ]

    print("Embedding 3 test texts...")
    vecs = embed_texts(test_texts)
    print(f"Shape: {vecs.shape}")  # (3, 384)

    # Similarity check
    from numpy import dot
    sim_ab = dot(vecs[0], vecs[1])
    sim_ac = dot(vecs[0], vecs[2])
    print(f"A-B similarity (both optimizers): {sim_ab:.3f}")
    print(f"A-C similarity (optimizer vs eval): {sim_ac:.3f}")
    print("A-B should be higher than A-C!")