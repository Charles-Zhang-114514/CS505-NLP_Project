import argparse
import datetime
import json
import os
import platform
import random
import re
import socket
import sys
import time
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src", "retrieval"))

from datasets import load_dataset

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

from src.data_prep.squad_loader import load_squad_qa_examples
from src.eval.qa_metrics import exact_match, f1_score
from src.generation.generator import SimpleGenerator, QwenGenerator
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.local_dense_retriever import LocalDenseRetriever
from src.retrieval.dense_retriever import DenseRetriever


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def contains_any_answer(text: str, answers: list[str]) -> str | None:
    text = normalize(text)
    for a in answers:
        ans = normalize(a)
        if not ans:
            continue
        if ans.isdigit():
            pattern = rf"(?<!\d){re.escape(ans)}(?!\d)"
            if re.search(pattern, text):
                return a
        else:
            pattern = rf"\b{re.escape(ans)}\b"
            if re.search(pattern, text):
                return a
    return None


def load_qa_examples(qa_dataset: str, split: str, num_questions: int, qa_path: str | None = None) -> list[dict[str, Any]]:
    if qa_path:
        with open(qa_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        return examples[:num_questions]

    if qa_dataset == "nq_open":
        dataset = load_dataset("nq_open", split=f"{split}[:{num_questions}]")
        examples = []
        for ex in dataset:
            answers = ex["answer"]
            if not answers:
                continue
            examples.append({"question": ex["question"], "answers": answers})
        return examples

    if qa_dataset == "squad":
        return load_squad_qa_examples(split=split, sample_size=num_questions)

    raise ValueError(f"Unsupported qa_dataset={qa_dataset}")


def load_chunks(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def best_em_f1_containment(prediction: str, gold_answers: list[str]) -> tuple[int, float, int]:
    em = max(exact_match(prediction, gold) for gold in gold_answers)
    f1 = max(f1_score(prediction, gold) for gold in gold_answers)
    containment = int(contains_any_answer(prediction, gold_answers) is not None)
    return em, f1, containment


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


def build_generator(generator_type: str, generator_model: str) -> Any:
    if generator_type == "simple":
        return SimpleGenerator(model_name=generator_model)

    if generator_type == "qwen":
        return QwenGenerator(model_name=generator_model)

    raise ValueError(f"Unsupported generator_type={generator_type}")


def detect_device(generator: Any) -> str:
    model = getattr(generator, "model", None)
    device = getattr(model, "device", None)
    if device is not None:
        return str(device)
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate_answer(generator: Any, generator_type: str, query: str, retrieved_chunks: list[dict[str, Any]], max_new_tokens: int) -> str:
    if generator_type == "simple":
        if not retrieved_chunks:
            return generator.answer_question(query)
        context_docs = [c["text"] for c in sorted(retrieved_chunks, key=lambda x: x.get("rank", 9999))]
        return generator.answer_with_context(query, context_docs)

    if generator_type == "qwen":
        if not retrieved_chunks:
            return generator.answer_closed_book(query, max_new_tokens=max_new_tokens)["answer"]
        return generator.generate(query, retrieved_chunks, max_new_tokens=max_new_tokens)["answer"]

    raise ValueError(f"Unsupported generator_type={generator_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified experiment runner for CS505 RAG project.")
    parser.add_argument("--mode", choices=["closed_book", "bm25", "local_dense", "qdrant_dense"], required=True)

    parser.add_argument("--qa_dataset", choices=["nq_open", "squad"], default="nq_open")
    parser.add_argument("--qa_split", default="validation")
    parser.add_argument("--qa_path", default=None, help="Optional local QA json file. If set, it overrides dataset loading.")
    parser.add_argument("--num_questions", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=5)

    parser.add_argument("--chunks_path", default=None, help="Required for bm25 and local_dense.")
    parser.add_argument("--embeddings_path", default=None, help="Required for local_dense.")
    parser.add_argument("--collection_name", default=None, help="Optional override for qdrant_dense.")
    parser.add_argument("--embedding_model", default="BAAI/bge-small-en-v1.5")

    parser.add_argument("--generator_type", choices=["simple", "qwen"], default="simple")
    parser.add_argument("--generator_model", default="google/flan-t5-base")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--corpus_name", default=None, help="Optional human-readable corpus label for metadata.")
    parser.add_argument("--chunking", default=None, help="Optional chunking label for metadata.")

    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--output_name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_started_at = utc_now_iso()
    run_start = time.perf_counter()
    set_global_seed(args.seed)

    setup_start = time.perf_counter()
    qa_examples = load_qa_examples(args.qa_dataset, args.qa_split, args.num_questions, args.qa_path)

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
    total_containment = 0.0
    generator = build_generator(
        generator_type=args.generator_type,
        generator_model=args.generator_model,
    )
    setup_time_sec = time.perf_counter() - setup_start
    total_retrieval_time = 0.0
    total_generation_time = 0.0
    total_example_time = 0.0

    for i, ex in enumerate(qa_examples, start=1):
        example_start = time.perf_counter()
        query = ex["question"]
        gold_answers = ex["answers"]

        retrieval_start = time.perf_counter()
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
        retrieval_time_sec = time.perf_counter() - retrieval_start

        generation_start = time.perf_counter()
        pred = generate_answer(
            generator=generator,
            generator_type=args.generator_type,
            query=query,
            retrieved_chunks=retrieved,
            max_new_tokens=args.max_new_tokens,
        )
        generation_time_sec = time.perf_counter() - generation_start
        example_time_sec = time.perf_counter() - example_start

        em, f1, containment = best_em_f1_containment(pred, gold_answers)
        total_em += em
        total_f1 += f1
        total_containment += containment
        total_retrieval_time += retrieval_time_sec
        total_generation_time += generation_time_sec
        total_example_time += example_time_sec

        results.append(
            {
                "index": i,
                "query": query,
                "gold_answers": gold_answers,
                "prediction": pred,
                "exact_match": em,
                "f1_score": f1,
                "answer_containment": containment,
                "retrieval_time_sec": round(retrieval_time_sec, 6),
                "generation_time_sec": round(generation_time_sec, 6),
                "example_time_sec": round(example_time_sec, 6),
                "retrieved_chunks": retrieved,
            }
        )

        print(f"[{i}/{len(qa_examples)}] {query}")
        print(f"  Pred: {pred}")
        print(f"  Gold: {gold_answers}")
        print(f"  EM: {em:.2f}  F1: {f1:.2f}  Containment: {containment}")
        print(
            f"  Time: retrieval={retrieval_time_sec:.3f}s  "
            f"generation={generation_time_sec:.3f}s  total={example_time_sec:.3f}s"
        )

    experiment_config = {
        "qa_dataset": args.qa_dataset,
        "qa_split": args.qa_split,
        "qa_path": args.qa_path,
        "num_questions": len(qa_examples),
        "mode": args.mode,
        "top_k": args.top_k,
        "chunks_path": args.chunks_path,
        "embeddings_path": args.embeddings_path,
        "collection_name": args.collection_name,
        "embedding_model": args.embedding_model,
        "generator_type": args.generator_type,
        "generator_model": args.generator_model,
        "max_new_tokens": args.max_new_tokens,
        "corpus_name": args.corpus_name,
        "chunking": args.chunking,
        "seed": args.seed,
    }
    run_finished_at = utc_now_iso()
    total_runtime_sec = time.perf_counter() - run_start
    runtime = {
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "setup_time_sec": round(setup_time_sec, 6),
        "total_runtime_sec": round(total_runtime_sec, 6),
        "avg_example_time_sec": round(total_example_time / len(qa_examples), 6) if qa_examples else 0.0,
        "avg_retrieval_time_sec": round(total_retrieval_time / len(qa_examples), 6) if qa_examples else 0.0,
        "avg_generation_time_sec": round(total_generation_time / len(qa_examples), 6) if qa_examples else 0.0,
    }
    environment = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "device": detect_device(generator),
    }

    summary = {
        "mode": args.mode,
        "qa_dataset": args.qa_dataset,
        "qa_split": args.qa_split,
        "num_questions": len(qa_examples),
        "top_k": args.top_k,
        "embedding_model": args.embedding_model if args.mode in ["local_dense", "qdrant_dense"] else None,
        "generator_type": args.generator_type,
        "generator_model": args.generator_model,
        "avg_exact_match": total_em / len(qa_examples) if qa_examples else 0.0,
        "avg_f1": total_f1 / len(qa_examples) if qa_examples else 0.0,
        "avg_answer_containment": total_containment / len(qa_examples) if qa_examples else 0.0,
        "total_runtime_sec": runtime["total_runtime_sec"],
        "avg_example_time_sec": runtime["avg_example_time_sec"],
        "avg_retrieval_time_sec": runtime["avg_retrieval_time_sec"],
        "avg_generation_time_sec": runtime["avg_generation_time_sec"],
        "runtime": runtime,
        "environment": environment,
        "experiment_config": experiment_config,
        "results": results,
    }

    ensure_dir(args.output_dir)
    output_name = args.output_name or f"{args.mode}_{args.qa_dataset}.json"
    output_path = os.path.join(args.output_dir, output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print(f"Saved results to: {output_path}")
    print(f"Avg EM: {summary['avg_exact_match']:.4f}")
    print(f"Avg F1: {summary['avg_f1']:.4f}")
    print(f"Avg Containment: {summary['avg_answer_containment']:.4f}")
    print(f"Total runtime: {summary['total_runtime_sec']:.2f}s")


if __name__ == "__main__":
    main()
