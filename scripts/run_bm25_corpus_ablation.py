# scripts/run_bm25_corpus_ablation.py
# Re-run BM25+RAG on ag_news and SQuAD corpora
# using NQ Open validation[:20] for fair comparison with Wiki oracle.

import json
import os
import sys
from datasets import load_dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from retrieval.bm25_retriever import BM25Retriever
from generation.generator import SimpleGenerator
from eval.qa_metrics import exact_match, f1_score


def load_nq_open_sample(split="validation", sample_size=20):
    dataset = load_dataset("nq_open", split=f"{split}[:{sample_size}]")
    return [{"question": ex["question"], "answers": ex["answer"]} for ex in dataset]


def load_ag_news_docs(sample_size=50):
    dataset = load_dataset("ag_news", split=f"train[:{sample_size}]")
    return [doc["text"] for doc in dataset]


def load_squad_docs(sample_size=1000, seed=42):
    dataset = load_dataset("squad", split="train")
    dataset = dataset.shuffle(seed=seed).select(range(sample_size))
    seen = set()
    docs = []
    for ex in dataset:
        text = ex["context"]
        if text not in seen:
            seen.add(text)
            docs.append(text)
    return docs


def max_em(pred, golds):
    return max(exact_match(pred, g) for g in golds)


def max_f1(pred, golds):
    return max(f1_score(pred, g) for g in golds)


def run_one(corpus_name, documents, examples, generator):
    retriever = BM25Retriever(documents)
    total_em = 0.0
    total_f1 = 0.0
    results = []

    for i, ex in enumerate(examples, start=1):
        q = ex["question"]
        golds = ex["answers"]
        retrieved = retriever.retrieve(q, top_k=2)
        pred = generator.answer_with_context(q, retrieved)
        em = max_em(pred, golds)
        f1 = max_f1(pred, golds)
        total_em += em
        total_f1 += f1
        results.append({
            "question": q, "gold_answers": golds,
            "prediction": pred, "em": em, "f1": f1
        })
        print(f"[{corpus_name}] {i}/{len(examples)}  EM={em}  F1={f1:.4f}")

    n = len(examples)
    avg_em = total_em / n
    avg_f1 = total_f1 / n
    print(f"\n===== {corpus_name} Final =====")
    print(f"Average EM: {avg_em:.4f}")
    print(f"Average F1: {avg_f1:.4f}")

    out_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"bm25_{corpus_name}_val20_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return avg_em, avg_f1


def main():
    print("Loading NQ Open validation[:20]...")
    examples = load_nq_open_sample()

    print("Loading generator...")
    generator = SimpleGenerator(model_name="google/flan-t5-base")

    # --- ag_news ---
    print("\n--- Loading ag_news corpus ---")
    ag_docs = load_ag_news_docs(sample_size=50)
    print(f"ag_news: {len(ag_docs)} docs")
    ag_em, ag_f1 = run_one("ag_news", ag_docs, examples, generator)

    # --- SQuAD ---
    print("\n--- Loading SQuAD corpus ---")
    sq_docs = load_squad_docs(sample_size=1000)
    print(f"squad: {len(sq_docs)} docs")
    sq_em, sq_f1 = run_one("squad", sq_docs, examples, generator)

    # --- Summary ---
    print("\n===== COMPARISON =====")
    print(f"  ag_news:  EM={ag_em:.4f}  F1={ag_f1:.4f}")
    print(f"  squad:    EM={sq_em:.4f}  F1={sq_f1:.4f}")


if __name__ == "__main__":
    main()