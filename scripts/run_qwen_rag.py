"""Stage 4 generation runner for Qwen-based QA evaluation."""

import json
from src.generation.generator import QwenGenerator
from src.eval.qa_metrics import exact_match, f1_score


def run(
    retrieved_results_path: str,
    output_path: str,
    model_name: str = "Qwen/Qwen3.5-4B",
    max_new_tokens: int = 64,
    mode: str = "rag",
):
    with open(retrieved_results_path, "r") as f:
        dataset = json.load(f)

    gen = QwenGenerator(model_name=model_name)

    results = []
    total_em, total_f1 = 0.0, 0.0

    for i, item in enumerate(dataset):
        query = item["query"]
        gold_answers = item["gold_answers"]
        retrieved_chunks = item.get("retrieved_chunks", [])

        if mode == "rag":
            result = gen.generate(query, retrieved_chunks, max_new_tokens)
        else:
            result = gen.answer_closed_book(query, max_new_tokens)

        pred = result["answer"]

        em = exact_match(pred, gold_answers)
        f1 = f1_score(pred, gold_answers)
        total_em += em
        total_f1 += f1

        result["gold_answers"] = gold_answers
        result["exact_match"] = em
        result["f1_score"] = f1
        results.append(result)

        print(f"[{i+1}/{len(dataset)}] Q: {query}")
        print(f"  Pred: {pred}")
        print(f"  Gold: {gold_answers}")
        print(f"  EM: {em:.2f}  F1: {f1:.2f}\n")

    n = len(dataset)
    summary = {
        "mode": mode,
        "model": model_name,
        "total": n,
        "avg_exact_match": total_em / n,
        "avg_f1": total_f1 / n,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 40)
    print(f"Avg EM : {summary['avg_exact_match']:.4f}")
    print(f"Avg F1 : {summary['avg_f1']:.4f}")
    print(f"Saved  : {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to the Stage 3 retrieval-results JSON file")
    parser.add_argument("--output", required=True, help="Path to save the evaluation results")
    parser.add_argument("--model",  default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--mode", choices=["rag", "closed_book"], default="rag")
    args = parser.parse_args()

    run(args.input, args.output, args.model, args.max_tokens, args.mode)
