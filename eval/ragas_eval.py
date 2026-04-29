"""Ragas metrikleri — Ollama ile local LLM kullanarak eval."""

from ragas import evaluate
from ragas.metrics import answer_relevancy
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

from src.config import OLLAMA_BASE_URL, LLM_MODEL


def get_ragas_llm():
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


def get_ragas_embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model="nomic-embed-text",
        openai_api_key="ollama",
        openai_api_base=f"{OLLAMA_BASE_URL}/v1",
    )


def run_ragas_eval(samples: list[dict]) -> dict:
    print(f"Running Ragas eval on {len(samples)} samples...")

    ragas_samples = []
    for s in samples:
        ragas_samples.append(SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s["reference"],
        ))

    dataset = EvaluationDataset(samples=ragas_samples)

    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    answer_relevancy.strictness = 1
    metrics = [answer_relevancy]

    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    results = evaluate(dataset=dataset, metrics=metrics)

    return {
        "answer_relevancy": results["answer_relevancy"],
    }


def print_results(scores: dict) -> None:
    print("\n" + "="*50)
    print("  RAGAS EVALUATION RESULTS")
    print("="*50)
    for metric, score in scores.items():
        if isinstance(score, list):
            valid = [s for s in score if s is not None and s == s]
            score = sum(valid) / len(valid) if valid else None
        if score is None or score != score:
            print(f"  {metric:<25} N/A — model structured output üretemedi")
            continue
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric:<25} {bar} {score:.3f}")
    print("="*50)