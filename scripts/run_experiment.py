import argparse
import json
import os
import sys
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src", "retrieval"))

from datasets import load_dataset

from src.eval.qa_metrics import exact_match, f1_score
from src.generation.generator import SimpleGenerator, QwenGenerator
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.local_dense_retriever import LocalDenseRetriever
from src.retrieval.dense_retriever import DenseRetriever


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_qa_examples(split: str, num_questions: int) -> list[dict[str, Any]]:
    dataset = load_dataset("nq_open", split=f"{split}[:{num_questions}]")
    examples = []
    for ex in dataset:
        answers = ex["answer"]
        if not answers:
            continue
        examples.append({"question": ex["question"], "answers": answers})
    return examples


def load_chunks(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def best_em_f1(prediction: str, gold_answers: list[str]) -> tuple[int, float]:
    em = max(exact_match(prediction, gold) for gold in gold_answers)
    f1 = max(f1_score(prediction, gold) for gold in gold_answers)
    return em, f1


def bm25_retrieve(chunks: list[dict[str, Any]], query: str, top_k: int) -> list[dict[str, Any]]:
    texts = [c["text"] for c in chunks]
    retriever = BM25Retriever(texts)
    scores = retriever.bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(ranked, start=1):
        item = dict(chunks[idx])
        item["score"] = float(scores[idx])
        item["rank"] = rank
        results.append(item)
    return results


def generate_answer(generator_type: str, generator_model: str, query: str, retrieved_chunks: list[dict[str, Any]], max_new_tokens: int) -> str:
    if generator_type == "simple":
        gen = SimpleGenerator(model_name=generator_model)
        if not retrieved_chunks:
            return gen.answer_question(query)
        context_docs = [c["text"] for c in sorted(retrieved_chunks, key=lambda x: x.get("rank", 9999))]
        return gen.answer_with_context(query, context_docs)

    if generator_type == "qwen":
        gen = QwenGenerator(model_name=generator_model)
        if not retrieved_chunks:
            return gen.answer_closed_book(query, max_new_tokens=max_new_tokens)["answer"]
        return gen.generate(query, retrieved_chunks, max_new_tokens=max_new_tokens)["answer"]

    raise ValueError(f"Unsupported generator_type={generator_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified experiment runner for CS505 RAG project.")
    parser.add_argument("--mode", choices=["closed_book", "bm25", "local_dense", "qdrant_dense"], required=True)
    parser.add_argument("--qa_split", default="validation")
    parser.add_argument("--num_questions", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=5)

    parser.add_argument("--chunks_path", default=None, help="Required for bm25 and local_dense.")
    parser.add_argument("--embeddings_path", default=None, help="Required for local_dense.")
    parser.add_argument("--collection_name", default=None, help="Optional override for qdrant_dense.")
    parser.add_argument("--embedding_model", default="BAAI/bge-small-en-v1.5")

    parser.add_argument("--generator_type", choices=["simple", "qwen"], default="simple")
    parser.add_argument("--generator_model", default="google/flan-t5-base")
    parser.add_argument("--max_new_tokens", type=int, default=64)

    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--output_name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qa_examples = load_qa_examples(args.qa_split, args.num_questions)

    chunks = None
    local_dense = None
    qdrant_dense = None

    if args.mode == "bm25":
        if not args.chunks_path:
            raise ValueError("--chunks_path is required for bm25")
        chunks = load_chunks(args.chunks_path)
    elif args.mode == "local_dense":
        if not args.chunks_path or not args.embeddings_path:
            raise ValueError("--chunks_path and --embeddings_path are required for local_dense")
        local_dense = LocalDenseRetriever(
            chunks_path=args.chunks_path,
            embeddings_path=args.embeddings_path,
            model_name=args.embedding_model,
        )
    elif args.mode == "qdrant_dense":
        qdrant_dense = DenseRetriever(
            collection_name=args.collection_name,
            model_name=args.embedding_model,
        )

    results = []
    total_em = 0.0
    total_f1 = 0.0

    for i, ex in enumerate(qa_examples, start=1):
        query = ex["question"]
        gold_answers = ex["answers"]

        if args.mode == "closed_book":
            retrieved = []
        elif args.mode == "bm25":
            retrieved = bm25_retrieve(chunks, query, args.top_k)
        elif args.mode == "local_dense":
            retrieved = local_dense.retrieve(query, k=args.top_k)
        elif args.mode == "qdrant_dense":
            retrieved = qdrant_dense.retrieve(query, k=args.top_k)
            for rank, hit in enumerate(retrieved, start=1):
                hit["rank"] = rank
        else:
            raise ValueError(f"Unsupported mode={args.mode}")

        pred = generate_answer(
            generator_type=args.generator_type,
            generator_model=args.generator_model,
            query=query,
            retrieved_chunks=retrieved,
            max_new_tokens=args.max_new_tokens,
        )

        em, f1 = best_em_f1(pred, gold_answers)
        total_em += em
        total_f1 += f1

        results.append(
            {
                "index": i,
                "query": query,
                "gold_answers": gold_answers,
                "prediction": pred,
                "exact_match": em,
                "f1_score": f1,
                "retrieved_chunks": retrieved,
            }
        )

        print(f"[{i}/{len(qa_examples)}] {query}")
        print(f"  Pred: {pred}")
        print(f"  Gold: {gold_answers}")
        print(f"  EM: {em:.2f}  F1: {f1:.2f}\n")

    summary = {
        "mode": args.mode,
        "qa_split": args.qa_split,
        "num_questions": len(qa_examples),
        "top_k": args.top_k,
        "embedding_model": args.embedding_model if args.mode in ["local_dense", "qdrant_dense"] else None,
        "generator_type": args.generator_type,
        "generator_model": args.generator_model,
        "avg_exact_match": total_em / len(qa_examples) if qa_examples else 0.0,
        "avg_f1": total_f1 / len(qa_examples) if qa_examples else 0.0,
        "results": results,
    }

    ensure_dir(args.output_dir)
    output_name = args.output_name or f"{args.mode}_nq_open.json"
    output_path = os.path.join(args.output_dir, output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print(f"Saved results to: {output_path}")
    print(f"Avg EM: {summary['avg_exact_match']:.4f}")
    print(f"Avg F1: {summary['avg_f1']:.4f}")


if __name__ == "__main__":
    main()
