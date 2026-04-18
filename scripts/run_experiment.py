import os
import sys
import json
import argparse
from typing import List, Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
# When this file is copied into scripts/, PROJECT_ROOT should be adjusted to ..
if os.path.basename(PROJECT_ROOT) == "scripts":
    PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src", "retrieval"))

from src.generation.generator import SimpleGenerator, QwenGenerator
from src.eval.qa_metrics import exact_match, f1_score
from src.retrieval.bm25_retriever import BM25Retriever

try:
    from src.retrieval.dense_retriever import DenseRetriever
except Exception:
    DenseRetriever = None

from src.chunking.fixed_chunk import fixed_chunk_document
from src.chunking.semantic_chunk import semantic_chunk_document

from scripts.load_nq import load_nq_open_sample
from scripts.load_wiki import load_doc_sample
from scripts.load_wiki_oracle import load_wiki_oracle_docs
from scripts.load_squad_docs import load_squad_docs


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_documents(raw_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs = []
    for i, doc in enumerate(raw_docs):
        docs.append(
            {
                "doc_id": doc.get("doc_id", f"doc_{i}"),
                "title": doc.get("title", ""),
                "text": doc.get("text", "").strip(),
            }
        )
    return [d for d in docs if d["text"]]


def load_qa_examples(dataset: str, split: str, sample_size: int) -> List[Dict[str, Any]]:
    if dataset != "nq_open":
        raise ValueError(f"Unsupported QA dataset: {dataset}")

    raw = load_nq_open_sample(split=split, sample_size=sample_size)
    examples = []
    for ex in raw:
        answers = ex.get("answers", [])
        if not answers:
            continue
        examples.append(
            {
                "question": ex["question"],
                "answers": answers,
            }
        )
    return examples


def load_corpus(corpus_name: str, sample_size: int) -> List[Dict[str, Any]]:
    if corpus_name == "ag_news":
        docs = load_doc_sample(sample_size=sample_size)
    elif corpus_name == "squad":
        docs = load_squad_docs(sample_size=sample_size)
    elif corpus_name == "wiki_oracle":
        docs = load_wiki_oracle_docs(nq_sample_size=sample_size)
    else:
        raise ValueError(
            f"Unsupported corpus: {corpus_name}. Supported: ag_news, squad, wiki_oracle"
        )
    return normalize_documents(docs)


def build_chunks(
    documents: List[Dict[str, Any]],
    chunking: str,
    fixed_chunk_size: int,
    fixed_overlap: int,
    semantic_chunk_size: int,
    semantic_threshold: float,
) -> List[Dict[str, Any]]:
    if chunking == "raw":
        chunks = []
        for doc in documents:
            chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_chunk_0",
                    "doc_id": doc["doc_id"],
                    "title": doc.get("title", ""),
                    "text": doc["text"],
                    "chunk_index": 0,
                    "method": "raw",
                }
            )
        return chunks

    all_chunks = []
    for doc in documents:
        if chunking == "fixed":
            doc_chunks = fixed_chunk_document(
                doc_id=doc["doc_id"],
                title=doc.get("title", ""),
                text=doc["text"],
                chunk_size=fixed_chunk_size,
                overlap=fixed_overlap,
            )
        elif chunking == "semantic":
            doc_chunks = semantic_chunk_document(
                doc_id=doc["doc_id"],
                title=doc.get("title", ""),
                text=doc["text"],
                chunk_size=semantic_chunk_size,
                threshold=semantic_threshold,
            )
        else:
            raise ValueError(f"Unsupported chunking method: {chunking}")
        all_chunks.extend(doc_chunks)

    return all_chunks


def best_em_f1(prediction: str, gold_answers: List[str]) -> tuple[int, float]:
    em = max(exact_match(prediction, gold) for gold in gold_answers)
    f1 = max(f1_score(prediction, gold) for gold in gold_answers)
    return em, f1


def bm25_retrieve(chunks: List[Dict[str, Any]], query: str, top_k: int) -> List[Dict[str, Any]]:
    texts = [c["text"] for c in chunks]
    retriever = BM25Retriever(texts)
    scores = retriever.bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(ranked, start=1):
        chunk = chunks[idx]
        item = dict(chunk)
        item["score"] = float(scores[idx])
        item["rank"] = rank
        results.append(item)
    return results


def dense_retrieve(query: str, top_k: int) -> List[Dict[str, Any]]:
    if DenseRetriever is None:
        raise RuntimeError(
            "DenseRetriever import failed. Check qdrant/sentence-transformers setup and imports."
        )
    retriever = DenseRetriever()
    hits = retriever.retrieve(query, k=top_k)
    for rank, hit in enumerate(hits, start=1):
        hit["rank"] = rank
    return hits


def generate_answer(
    generator_type: str,
    generator_model: str,
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    max_new_tokens: int,
) -> str:
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

    raise ValueError(f"Unsupported generator_type: {generator_type}")


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    qa_examples = load_qa_examples(args.qa_dataset, args.qa_split, args.qa_sample_size)

    corpus_docs = []
    chunks = []
    if args.mode != "closed_book":
        corpus_docs = load_corpus(args.corpus, args.corpus_sample_size)
        chunks = build_chunks(
            documents=corpus_docs,
            chunking=args.chunking,
            fixed_chunk_size=args.fixed_chunk_size,
            fixed_overlap=args.fixed_overlap,
            semantic_chunk_size=args.semantic_chunk_size,
            semantic_threshold=args.semantic_threshold,
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
        elif args.mode == "dense":
            retrieved = dense_retrieve(query, args.top_k)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

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

        result = {
            "index": i,
            "query": query,
            "gold_answers": gold_answers,
            "prediction": pred,
            "exact_match": em,
            "f1_score": f1,
            "retrieved_chunks": retrieved,
        }
        results.append(result)

        print(f"[{i}/{len(qa_examples)}] {query}")
        print(f"  Pred: {pred}")
        print(f"  Gold: {gold_answers}")
        print(f"  EM: {em:.2f}  F1: {f1:.2f}")
        if retrieved:
            print(f"  Retrieved: {len(retrieved)} chunks")
        print()

    summary = {
        "mode": args.mode,
        "qa_dataset": args.qa_dataset,
        "qa_split": args.qa_split,
        "qa_sample_size": len(qa_examples),
        "corpus": args.corpus if args.mode != "closed_book" else None,
        "corpus_sample_size": len(corpus_docs) if args.mode != "closed_book" else 0,
        "chunking": args.chunking if args.mode != "closed_book" else None,
        "num_chunks": len(chunks),
        "top_k": args.top_k if args.mode != "closed_book" else 0,
        "generator_type": args.generator_type,
        "generator_model": args.generator_model,
        "avg_exact_match": total_em / len(qa_examples) if qa_examples else 0.0,
        "avg_f1": total_f1 / len(qa_examples) if qa_examples else 0.0,
        "results": results,
    }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified experiment entry point for CS505 RAG project.")

    parser.add_argument("--mode", choices=["closed_book", "bm25", "dense"], required=True)

    parser.add_argument("--qa_dataset", default="nq_open", choices=["nq_open"])
    parser.add_argument("--qa_split", default="validation")
    parser.add_argument("--qa_sample_size", type=int, default=20)

    parser.add_argument("--corpus", default="wiki_oracle", choices=["ag_news", "squad", "wiki_oracle"])
    parser.add_argument("--corpus_sample_size", type=int, default=20)

    parser.add_argument("--chunking", default="raw", choices=["raw", "fixed", "semantic"])
    parser.add_argument("--top_k", type=int, default=5)

    parser.add_argument("--fixed_chunk_size", type=int, default=120)
    parser.add_argument("--fixed_overlap", type=int, default=20)
    parser.add_argument("--semantic_chunk_size", type=int, default=250)
    parser.add_argument("--semantic_threshold", type=float, default=0.75)

    parser.add_argument("--generator_type", default="simple", choices=["simple", "qwen"])
    parser.add_argument("--generator_model", default="google/flan-t5-base")
    parser.add_argument("--max_new_tokens", type=int, default=64)

    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--output_name", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(args)

    ensure_dir(args.output_dir)
    if args.output_name is None:
        suffix = f"{args.mode}_{args.qa_dataset}"
        if args.mode != "closed_book":
            suffix += f"_{args.corpus}_{args.chunking}"
        output_name = suffix + ".json"
    else:
        output_name = args.output_name

    output_path = os.path.join(args.output_dir, output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print(f"Saved results to: {output_path}")
    print(f"Avg EM: {summary['avg_exact_match']:.4f}")
    print(f"Avg F1: {summary['avg_f1']:.4f}")


if __name__ == "__main__":
    main()
