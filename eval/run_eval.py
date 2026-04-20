"""CLI entry point for offline evaluation.

Kullanım: python -m eval.run_eval --samples 20
"""

import argparse
import json
from src.config import DATA_DIR
from eval.dataset_builder import build_eval_dataset
from eval.ragas_eval import run_ragas_eval, print_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    # 1. Dataset oluştur
    samples = build_eval_dataset(max_samples=args.samples)

    # 2. Ragas eval çalıştır
    scores = run_ragas_eval(samples)
    print_results(scores)

    # 3. Kaydet
    if args.save:
        out_path = DATA_DIR / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()