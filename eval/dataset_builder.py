"""QASPER golden QA pair'lerinden Ragas eval dataset'i oluştur.

Her satır: question, answer (LLM), contexts (retrieved), reference (golden answer)
"""

from src.data_loader import load_from_disk
from src.vector_store import FAISSStore
from src.generator import generate
from src.config import TOP_K


def build_eval_dataset(max_samples: int = 20, top_k: int = TOP_K) -> list[dict]:
    """RAG pipeline'ı golden QA pair'ler üzerinde çalıştır.

    Args:
        max_samples: Kaç soru denensin (CPU yavaş, 20 yeterli).
        top_k: Kaç chunk retrieve edilsin.

    Returns:
        List of dicts: question, answer, contexts, reference
    """
    qa_pairs = load_from_disk("qasper_qa_pairs.jsonl")[:max_samples]
    store = FAISSStore.load()

    samples = []
    print(f"Building eval dataset — {len(qa_pairs)} questions...")

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        reference = qa["answer"]  # golden answer

        print(f"  [{i+1}/{len(qa_pairs)}] {question[:60]}...")

        # Retrieve
        results = store.search(question, top_k=top_k)
        contexts = [r["text"] for r in results]

        # Generate
        try:
            gen_result = generate(query=question, context_chunks=results)
            answer = gen_result["answer"]
        except Exception as e:
            print(f"    Generation failed: {e}")
            answer = ""

        samples.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "reference": reference,
        })

    return samples