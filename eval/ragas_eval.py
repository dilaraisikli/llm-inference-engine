"""Ragas metrikleri — Ollama ile local LLM ve embedding kullanarak eval."""

from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas import evaluate
#from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.metrics import answer_relevancy
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

from src.config import OLLAMA_BASE_URL, LLM_MODEL


def get_ragas_llm():
    """Ollama'yı Ragas LLM olarak yapılandır."""
    client = OpenAI(
        api_key="ollama",
        base_url=f"{OLLAMA_BASE_URL}/v1",
    )
    return llm_factory(LLM_MODEL, provider="openai", client=client)


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

    #metrics = [faithfulness, answer_relevancy, context_precision]
    metrics = [answer_relevancy]

    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    results = evaluate(dataset=dataset, metrics=metrics)

    return {
        #"faithfulness": results["faithfulness"],
        "answer_relevancy": results["answer_relevancy"],
        #"context_precision": results["context_precision"],
    }


def print_results(scores: dict) -> None:
    print("\n" + "="*50)
    print("  RAGAS EVALUATION RESULTS")
    print("="*50)
    for metric, score in scores.items():
        if isinstance(score, list):
            score = sum(s for s in score if s is not None) / max(len([s for s in score if s is not None]), 1)
        if score is None:
            print(f"  {metric:<25} N/A")
            continue
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric:<25} {bar} {score:.3f}")
    print("="*50)