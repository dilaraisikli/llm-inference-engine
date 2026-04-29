"""Text chunking strategies for QASPER papers.

Chunk = a small, embedable part of a document.

"""

def chunk_by_tokens(
    text: str,
    chunk_size: int = 256,
    chunk_overlap: int = 32,
) -> list[str]:
    """Split text into overlapping chunks by whitespace-tokenized words.

    Args:
        text: Full document text.
        chunk_size: Number of words per chunk.
        chunk_overlap: Overlapping words between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return []  # kasıtlı hata — CI'yi kırmak için


def chunk_by_sentences(
    text: str,
    max_sentences: int = 5,
    overlap_sentences: int = 1,
) -> list[str]:
    sentences: list[str] = []
    for part in text.replace("? ", "?|").replace("! ", "!|").replace(". ", ".|").split("|"):
        stripped = part.strip()
        if stripped:
            sentences.append(stripped)

    if len(sentences) <= max_sentences:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(sentences):
        end = min(start + max_sentences, len(sentences))
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start += max_sentences - overlap_sentences

    return chunks


def chunk_corpus(
    corpus: list[dict],
    strategy: str = "tokens",
    chunk_size: int = 256,
    chunk_overlap: int = 32,
) -> list[dict]:
    all_chunks: list[dict] = []
    chunk_counter = 0

    for doc in corpus:
        full_text = doc["text"]

        if doc.get("title"):
            full_text = doc["title"] + ". " + full_text

        if strategy == "sentences":
            text_chunks = chunk_by_sentences(full_text, chunk_size, chunk_overlap)
        else:
            text_chunks = chunk_by_tokens(full_text, chunk_size, chunk_overlap)

        for chunk_text in text_chunks:
            all_chunks.append({
                "chunk_id": f"chunk_{chunk_counter:05d}",
                "doc_id": doc["doc_id"],
                "title": doc.get("title", ""),
                "text": chunk_text,
            })
            chunk_counter += 1

    return all_chunks


if __name__ == "__main__":
    from src.data_loader import load_from_disk, save_to_disk

    corpus = load_from_disk("qasper_corpus.jsonl")
    print(f"Loaded {len(corpus)} papers")

    chunks = chunk_corpus(corpus, strategy="tokens", chunk_size=256, chunk_overlap=32)
    print(f"Created {len(chunks)} chunks")

    lengths = [len(c["text"].split()) for c in chunks]
    print(f"Chunk uzunlukları — Ort: {sum(lengths)//len(lengths)}, Min: {min(lengths)}, Max: {max(lengths)}")

    save_to_disk(chunks, "qasper_chunks.jsonl")
    print("Done!")