"""Download QASPER dataset from HuggingFace and save locally.

QASPER: 5,049 questions over 1,585 NLP papers.
Each paper has full_text (long), questions, answers, and evidence.
"""

import json
from pathlib import Path

from datasets import load_dataset

from src.config import DATA_DIR, QASPER_SUBSET_SIZE


def load_qasper(subset_size: int = QASPER_SUBSET_SIZE) -> tuple[list[dict], list[dict]]:
    """Load QASPER dataset.

    Returns:
        corpus: list of dicts with keys: doc_id, title, text
        qa_pairs: list of dicts with keys: query_id, doc_id, question, answer, evidence
    """
    
    dataset = load_dataset("allenai/qasper", "default", split="train", revision="refs/convert/parquet")
    corpus: list[dict] = []
    qa_pairs: list[dict] = []
    paper_count = 0

    for row in dataset:
        if paper_count >= subset_size:
            break

        paper_id = row["id"]
        title = row.get("title", "")

        # merge full_text parag.
        full_text_parts = []
        sections = row.get("full_text", {})
        section_names = sections.get("section_name", [])
        paragraphs = sections.get("paragraphs", [])

        for sec_name, sec_paragraphs in zip(section_names, paragraphs):
            if sec_name:
                full_text_parts.append(sec_name)
            for para in sec_paragraphs:
                if para:
                    full_text_parts.append(para)
            

        full_text = "\n".join(full_text_parts)

        # Corpus'a ekle
        corpus.append({
            "doc_id": paper_id,
            "title": title,
            "text": full_text,
        })

        # QA pair'leri çıkar
        qas = row.get("qas", {})
        questions = qas.get("question", [])
        question_ids = qas.get("question_id", [])
        answers_list = qas.get("answers", [])

        for q_text, q_id, ans_data in zip(questions, question_ids, answers_list):
            answer_objs = ans_data.get("answer", [])

            # İlk cevaplanabilir answer'ı al
            best_answer = ""
            evidence = []

            for ans in answer_objs:
                if ans.get("unanswerable", False):
                    continue

                # free_form_answer > extractive_spans > yes_no
                if ans.get("free_form_answer"):
                    best_answer = ans["free_form_answer"]
                elif ans.get("extractive_spans"):
                    best_answer = " ".join(ans["extractive_spans"])
                elif ans.get("yes_no") is not None:
                    best_answer = "Yes" if ans["yes_no"] else "No"

                evidence = [e for e in ans.get("evidence", []) if e and not e.startswith("FLOAT")]
                break

            # Sadece cevabı olan soruları al
            if best_answer:
                qa_pairs.append({
                    "query_id": q_id,
                    "doc_id": paper_id,
                    "question": q_text,
                    "answer": best_answer,
                    "evidence": evidence,
                })

        paper_count += 1

    return corpus, qa_pairs


def save_to_disk(data: list[dict], file_name: str, path: Path | None = None) -> Path:
    """Save data as JSONL."""
    out_path = path or DATA_DIR / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(data)} items to {out_path}")
    return out_path


def load_from_disk(file_name: str, path: Path | None = None) -> list[dict]:
    """Load previously saved data from JSONL."""
    in_path = path or DATA_DIR / file_name

    data: list[dict] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    return data


if __name__ == "__main__":
    print("Downloading QASPER...")
    corpus, qa_pairs = load_qasper()

    save_to_disk(corpus, "qasper_corpus.jsonl")
    print(f"Corpus: {len(corpus)} papers")

    # Kelime uzunluğu kontrolü
    lengths = [len(c["text"].split()) for c in corpus]
    print(f"Avg: {sum(lengths)//len(lengths)} words, Min: {min(lengths)}, Max: {max(lengths)}")

    save_to_disk(qa_pairs, "qasper_qa_pairs.jsonl")
    print(f"QA pairs: {len(qa_pairs)} items (golden set)")