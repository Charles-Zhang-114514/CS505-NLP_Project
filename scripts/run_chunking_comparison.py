import json
import os
import sys
from typing import Dict, List

# -------------------------
# Path setup
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.load_squad_qa import load_squad_qa_sample
from scripts.load_squad_docs import load_squad_docs

from src.chunking.fixed_chunk import fixed_chunk_documents
from src.chunking.semantic_chunk import semantic_chunk_documents
from src.retrieval.bm25_retriever import BM25Retriever
from src.generation.generator import SimpleGenerator
from src.eval.qa_metrics import exact_match, f1_score, normalize_text


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def max_em_over_answers(prediction: str, gold_answers: List[str]) -> float:
    return max(exact_match(prediction, ans) for ans in gold_answers)


def max_f1_over_answers(prediction: str, gold_answers: List[str]) -> float:
    return max(f1_score(prediction, ans) for ans in gold_answers)


def retrieval_hit_at_k(retrieved_docs: List[str], gold_answers: List[str]) -> int:
    normalized_docs = [normalize_text(doc) for doc in retrieved_docs]

    for ans in gold_answers:
        norm_ans = normalize_text(ans)
        if not norm_ans:
            continue
        for doc in normalized_docs:
            if norm_ans in doc:
                return 1
    return 0


def build_raw_units(documents: List[Dict]) -> List[Dict]:
    units = []
    for i, doc in enumerate(documents):
        units.append({
            "chunk_id": f"{doc['doc_id']}_raw_0",
            "doc_id": doc["doc_id"],
            "title": doc.get("title", ""),
            "text": doc["text"],
            "chunk_index": 0,
            "method": "raw",
        })
    return units


def build_retrieval_units(
    documents: List[Dict],
    strategy: str,
    fixed_chunk_size: int = 120,
    fixed_overlap: int = 20,
    semantic_chunk_size: int = 250,
    semantic_threshold: float = 0.75,
) -> List[Dict]:
    if strategy == "raw":
        return build_raw_units(documents)

    if strategy == "fixed":
        return fixed_chunk_documents(
            documents,
            chunk_size=fixed_chunk_size,
            overlap=fixed_overlap,
        )

    if strategy == "semantic":
        return semantic_chunk_documents(
            documents,
            chunk_size=semantic_chunk_size,
            threshold=semantic_threshold,
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_strategy(
    strategy: str,
    qa_examples: List[Dict],
    doc_objects: List[Dict],
    generator: SimpleGenerator,
    top_k: int = 2,
    fixed_chunk_size: int = 120,
    fixed_overlap: int = 20,
    semantic_chunk_size: int = 250,
    semantic_threshold: float = 0.75,
) -> Dict:
    retrieval_units = build_retrieval_units(
        documents=doc_objects,
        strategy=strategy,
        fixed_chunk_size=fixed_chunk_size,
        fixed_overlap=fixed_overlap,
        semantic_chunk_size=semantic_chunk_size,
        semantic_threshold=semantic_threshold,
    )

    unit_texts = [u["text"] for u in retrieval_units]
    retriever = BM25Retriever(unit_texts)

    total_em = 0.0
    total_f1 = 0.0
    total_hit = 0.0
    all_results = []

    for i, example in enumerate(qa_examples, start=1):
        question = example["question"]
        gold_answers = example["answers"]

        retrieved_docs = retriever.retrieve(question, top_k=top_k)
        prediction = generator.answer_with_context(question, retrieved_docs)

        em = max_em_over_answers(prediction, gold_answers)
        f1 = max_f1_over_answers(prediction, gold_answers)
        hit = retrieval_hit_at_k(retrieved_docs, gold_answers)

        total_em += em
        total_f1 += f1
        total_hit += hit

        all_results.append({
            "question": question,
            "gold_answers": gold_answers,
            "retrieved_docs": retrieved_docs,
            "prediction": prediction,
            "em": em,
            "f1": f1,
            "retrieval_hit@k": hit,
        })

        print(f"\n[{strategy}] Example {i}")
        print("Question:", question)
        print("Gold Answers:", gold_answers)
        print("Prediction:", prediction)
        print("EM:", em)
        print("F1:", round(f1, 4))
        print("Retrieval Hit@K:", hit)

    n = len(qa_examples)
    summary = {
        "strategy": strategy,
        "num_questions": n,
        "num_retrieval_units": len(retrieval_units),
        "avg_em": total_em / n,
        "avg_f1": total_f1 / n,
        "retrieval_hit_rate@k": total_hit / n,
        "top_k": top_k,
        "fixed_chunk_size": fixed_chunk_size,
        "fixed_overlap": fixed_overlap,
        "semantic_chunk_size": semantic_chunk_size,
        "semantic_threshold": semantic_threshold,
    }

    return {
        "summary": summary,
        "details": all_results,
    }


def main():
    # -------------------------
    # Config
    # -------------------------
    qa_sample_size = 20
    doc_sample_size = 300
    top_k = 2

    fixed_chunk_size = 120
    fixed_overlap = 20

    semantic_chunk_size = 250
    semantic_threshold = 0.75

    strategies = ["raw", "fixed", "semantic"]

    # -------------------------
    # Load data
    # -------------------------
    print("Loading QA examples...")
    qa_examples = load_squad_qa_sample(sample_size=qa_sample_size)

    print("Loading documents...")
    doc_objects = load_squad_docs(sample_size=doc_sample_size)

    # -------------------------
    # Same generator for all strategies
    # -------------------------
    print("Loading generator...")
    generator = SimpleGenerator(model_name="google/flan-t5-base")

    all_summaries = []
    all_outputs = {}

    for strategy in strategies:
        print("\n" + "=" * 60)
        print(f"Running strategy: {strategy}")
        print("=" * 60)

        result = evaluate_strategy(
            strategy=strategy,
            qa_examples=qa_examples,
            doc_objects=doc_objects,
            generator=generator,
            top_k=top_k,
            fixed_chunk_size=fixed_chunk_size,
            fixed_overlap=fixed_overlap,
            semantic_chunk_size=semantic_chunk_size,
            semantic_threshold=semantic_threshold,
        )

        all_outputs[strategy] = result
        all_summaries.append(result["summary"])

        out_path = os.path.join(
            PROJECT_ROOT,
            "results",
            f"chunking_{strategy}_results.json",
        )
        save_json(result, out_path)

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    for summary in all_summaries:
        print(
            f"{summary['strategy']:>8} | "
            f"units={summary['num_retrieval_units']:>5} | "
            f"EM={summary['avg_em']:.4f} | "
            f"F1={summary['avg_f1']:.4f} | "
            f"Hit@{summary['top_k']}={summary['retrieval_hit_rate@k']:.4f}"
        )

    save_json(
        all_summaries,
        os.path.join(PROJECT_ROOT, "results", "chunking_comparison_summary.json"),
    )


if __name__ == "__main__":
    main()