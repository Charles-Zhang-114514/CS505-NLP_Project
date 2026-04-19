import argparse
import csv
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from datasets import load_dataset

from src.data_prep.squad_loader import load_squad_qa_examples


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def token_len(text: str) -> int:
    return len(str(text).split())


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
            return json.load(f)[:num_questions]

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


def print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def maybe_export_csv(path: str | None, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    if not rows:
        print(f"No rows to export to {path}")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported CSV to {path}")


def sort_items(items: list[dict[str, Any]], sort_by: str, descending: bool) -> list[dict[str, Any]]:
    if sort_by == "index":
        key_fn = lambda x: x.get("index", 0)
    elif sort_by == "f1":
        key_fn = lambda x: x.get("f1_score", 0.0)
    elif sort_by == "em":
        key_fn = lambda x: x.get("exact_match", 0)
    elif sort_by == "containment":
        key_fn = lambda x: x.get("answer_containment", 0)
    else:
        key_fn = lambda x: x.get(sort_by, 0)
    return sorted(items, key=key_fn, reverse=descending)


def filter_result_items(items: list[dict[str, Any]], only_errors: bool, only_nonzero_f1: bool) -> list[dict[str, Any]]:
    out = []
    for item in items:
        if only_errors and item.get("exact_match", 0) == 1:
            continue
        if only_nonzero_f1 and item.get("f1_score", 0.0) == 0:
            continue
        out.append(item)
    return out


def metadata_from_result(data: dict[str, Any], result_path: str, corpus_path: str | None = None, chunks_path: str | None = None) -> dict[str, Any]:
    experiment_config = data.get("experiment_config", {})
    meta = {
        "result_file": result_path,
        "mode": data.get("mode"),
        "qa_dataset": data.get("qa_dataset") or experiment_config.get("qa_dataset"),
        "qa_split": data.get("qa_split") or experiment_config.get("qa_split"),
        "num_questions": data.get("num_questions") or experiment_config.get("num_questions"),
        "top_k": data.get("top_k") or experiment_config.get("top_k"),
        "generator_type": data.get("generator_type") or experiment_config.get("generator_type"),
        "generator_model": data.get("generator_model") or experiment_config.get("generator_model"),
        "embedding_model": data.get("embedding_model") or experiment_config.get("embedding_model"),
        "corpus_name": experiment_config.get("corpus_name"),
        "chunking": experiment_config.get("chunking"),
        "avg_em": data.get("avg_exact_match"),
        "avg_f1": data.get("avg_f1"),
        "avg_answer_containment": data.get("avg_answer_containment"),
    }
    if corpus_path:
        meta["corpus_path"] = corpus_path
    if chunks_path:
        meta["chunks_path"] = chunks_path
    return meta


def command_summary(args: argparse.Namespace) -> None:
    data = load_json(args.result_path)
    results = data["results"]

    print_header("Experiment Summary")
    meta = metadata_from_result(data, args.result_path, args.corpus_path, args.chunks_path)
    for k, v in meta.items():
        print(f"{k}: {v}")

    export_rows: list[dict[str, Any]] = []

    if args.chunks_path:
        chunks = load_json(args.chunks_path)
        lengths = [token_len(c["text"]) for c in chunks]
        print("\nChunk stats")
        print(f"num_chunks: {len(chunks)}")
        print(f"avg_chunk_len: {round(statistics.mean(lengths), 2) if lengths else 0}")
        print(f"median_chunk_len: {statistics.median(lengths) if lengths else 0}")
        print(f"min_chunk_len: {min(lengths) if lengths else 0}")
        print(f"max_chunk_len: {max(lengths) if lengths else 0}")
        export_rows.append({
            "section": "chunk_stats",
            "num_chunks": len(chunks),
            "avg_chunk_len": round(statistics.mean(lengths), 2) if lengths else 0,
            "median_chunk_len": statistics.median(lengths) if lengths else 0,
            "min_chunk_len": min(lengths) if lengths else 0,
            "max_chunk_len": max(lengths) if lengths else 0,
        })

    if args.corpus_path:
        qa_examples = load_qa_examples(args.qa_dataset, args.qa_split, args.num_questions, args.qa_path)
        corpus = load_json(args.corpus_path)
        all_text = "\n".join(doc["text"].lower() for doc in corpus)
        coverage = 0
        for ex in qa_examples:
            if contains_any_answer(all_text, ex["answers"]):
                coverage += 1
        print("\nCorpus coverage")
        print(f"coverage: {coverage}/{len(qa_examples)} = {coverage / len(qa_examples):.4f}")
        export_rows.append({
            "section": "coverage",
            "coverage_count": coverage,
            "coverage_total": len(qa_examples),
            "coverage_rate": coverage / len(qa_examples) if qa_examples else 0.0,
        })

    retrieval_hits = 0
    for item in results:
        if any(contains_any_answer(chunk["text"], item["gold_answers"]) for chunk in item.get("retrieved_chunks", [])):
            retrieval_hits += 1
    print("\nRetrieval hit@k")
    print(f"hit@k: {retrieval_hits}/{len(results)} = {retrieval_hits / len(results):.4f}")
    export_rows.append({
        "section": "retrieval",
        "hit_count": retrieval_hits,
        "hit_total": len(results),
        "hit_rate": retrieval_hits / len(results) if results else 0.0,
    })

    containment_hits = sum(item.get("answer_containment", 0) for item in results)
    print("\nFinal QA metrics")
    print(f"avg_em: {data.get('avg_exact_match')}")
    print(f"avg_f1: {data.get('avg_f1')}")
    print(f"avg_answer_containment: {data.get('avg_answer_containment')}")
    export_rows.append({
        "section": "final_metrics",
        "avg_em": data.get("avg_exact_match"),
        "avg_f1": data.get("avg_f1"),
        "avg_answer_containment": data.get("avg_answer_containment"),
        "containment_count": containment_hits,
        "containment_total": len(results),
    })

    maybe_export_csv(args.export_csv, export_rows)


def command_inspect(args: argparse.Namespace) -> None:
    data = load_json(args.result_path)
    items = data["results"]
    items = filter_result_items(items, args.only_errors, args.only_nonzero_f1)
    items = sort_items(items, args.sort_by, args.descending)
    if args.limit is not None and args.limit >= 0:
        items = items[:args.limit]

    print_header("Inspect Results")
    export_rows = []
    for item in items:
        print(f"Index: {item.get('index')}")
        print(f"Query: {item.get('query')}")
        print(f"Gold: {item.get('gold_answers')}")
        print(f"Prediction: {item.get('prediction')}")
        print(
            f"EM: {item.get('exact_match')}  F1: {round(item.get('f1_score', 0.0), 4)}  "
            f"Containment: {item.get('answer_containment', 0)}"
        )
        if args.show_retrieved:
            for j, chunk in enumerate(item.get("retrieved_chunks", []), start=1):
                print(f"  [{j}] chunk_id={chunk.get('chunk_id')} doc_id={chunk.get('doc_id')} score={chunk.get('score')}")
                if args.show_chunk_text:
                    print(f"      {chunk.get('text')}")
        print("-" * 80)
        export_rows.append({
            "index": item.get("index"),
            "query": item.get("query"),
            "gold_answers": " || ".join(item.get("gold_answers", [])),
            "prediction": item.get("prediction"),
            "exact_match": item.get("exact_match"),
            "f1_score": item.get("f1_score"),
            "answer_containment": item.get("answer_containment", 0),
        })
    maybe_export_csv(args.export_csv, export_rows)


def command_retrieval(args: argparse.Namespace) -> None:
    data = load_json(args.result_path)
    items = data["results"]
    hit_count = 0
    rows = []
    shown = 0

    print_header("Retrieval Analysis")
    for item in items:
        query = item["query"]
        gold_answers = item["gold_answers"]
        retrieved_chunks = item.get("retrieved_chunks", [])

        hit = False
        matched_answer = None
        matched_chunk = None
        for chunk in retrieved_chunks:
            ans = contains_any_answer(chunk["text"], gold_answers)
            if ans is not None:
                hit = True
                matched_answer = ans
                matched_chunk = chunk
                break

        if hit:
            hit_count += 1

        should_show = (hit and args.show_hits) or ((not hit) and args.show_misses)
        if should_show and (args.limit < 0 or shown < args.limit):
            tag = "HIT" if hit else "MISS"
            print(f"[{tag} {item.get('index')}]")
            print(f"Query: {query}")
            print(f"Gold answers: {gold_answers}")
            if hit:
                print(f"Matched answer: {matched_answer}")
                print(f"Matched chunk_id: {matched_chunk.get('chunk_id')}")
                print(f"Matched doc_id: {matched_chunk.get('doc_id')}")
                print(f"Matched score: {matched_chunk.get('score')}")
                if args.show_chunk_text:
                    print(f"Matched text: {matched_chunk.get('text')}")
            else:
                print(f"Top-k retrieved chunk_ids: {[c.get('chunk_id') for c in retrieved_chunks]}")
            print("-" * 80)
            shown += 1

        rows.append({
            "index": item.get("index"),
            "query": query,
            "gold_answers": " || ".join(gold_answers),
            "retrieval_hit": int(hit),
            "matched_answer": matched_answer,
            "matched_chunk_id": matched_chunk.get("chunk_id") if matched_chunk else None,
            "matched_doc_id": matched_chunk.get("doc_id") if matched_chunk else None,
            "matched_score": matched_chunk.get("score") if matched_chunk else None,
        })

    print(f"retrieval hit@k: {hit_count}/{len(items)} = {hit_count / len(items):.4f}")
    maybe_export_csv(args.export_csv, rows)


def command_diagnose(args: argparse.Namespace) -> None:
    corpus = load_json(args.corpus_path)
    data = load_json(args.result_path)
    results = data["results"]
    qa_examples = load_qa_examples(args.qa_dataset, args.qa_split, args.num_questions, args.qa_path)

    all_text = "\n".join(doc["text"].lower() for doc in corpus)
    counts = {
        "not_in_corpus": 0,
        "in_corpus_not_retrieved": 0,
        "retrieved_but_wrong_answer": 0,
        "retrieved_and_partially_correct": 0,
        "exact_match": 0,
    }
    groups = {k: [] for k in counts.keys()}

    for ex, item in zip(qa_examples, results):
        query = ex["question"]
        gold_answers = ex["answers"]
        pred = item["prediction"]
        em = item["exact_match"]
        f1 = item["f1_score"]
        retrieved = item.get("retrieved_chunks", [])

        in_corpus = contains_any_answer(all_text, gold_answers) is not None
        retrieved_hit = any(contains_any_answer(chunk["text"], gold_answers) for chunk in retrieved)

        if not in_corpus:
            label = "not_in_corpus"
        elif in_corpus and not retrieved_hit:
            label = "in_corpus_not_retrieved"
        elif retrieved_hit and em == 1:
            label = "exact_match"
        elif retrieved_hit and (f1 > 0 or item.get("answer_containment", 0) == 1):
            label = "retrieved_and_partially_correct"
        else:
            label = "retrieved_but_wrong_answer"

        counts[label] += 1
        groups[label].append({
            "index": item.get("index"),
            "query": query,
            "gold_answers": gold_answers,
            "prediction": pred,
            "exact_match": em,
            "f1_score": f1,
            "answer_containment": item.get("answer_containment", 0),
        })

    print_header("Pipeline Diagnosis")
    print("Summary")
    for k, v in counts.items():
        print(f"{k}: {v}/{len(qa_examples)}")

    rows = []
    for k, items in groups.items():
        print("\n" + "-" * 80)
        print(f"{k} (showing up to {args.limit if args.limit >= 0 else 'all'})")
        subset = items if args.limit < 0 else items[:args.limit]
        for item in subset:
            print(f"Q: {item['query']}")
            print(f"Gold: {item['gold_answers']}")
            print(f"Pred: {item['prediction']}")
            print(
                f"EM: {item['exact_match']} F1: {round(item['f1_score'], 4)} "
                f"Containment: {item['answer_containment']}"
            )
            print("-" * 80)
        for item in items:
            rows.append({"error_type": k, **item, "gold_answers": " || ".join(item["gold_answers"])})

    maybe_export_csv(args.export_csv, rows)


def command_compare(args: argparse.Namespace) -> None:
    print_header("Compare Experiments")
    rows = []
    for path in args.result_paths:
        data = load_json(path)
        row = metadata_from_result(data, path)
        rows.append(row)

    rows = sorted(rows, key=lambda x: (x.get(args.sort_by) is None, x.get(args.sort_by)), reverse=args.descending)
    if args.limit >= 0:
        rows = rows[:args.limit]

    for row in rows:
        print(f"file: {Path(row['result_file']).name}")
        print(f"  mode={row.get('mode')} generator={row.get('generator_type')} model={row.get('generator_model')}")
        print(f"  qa_dataset={row.get('qa_dataset')} qa_split={row.get('qa_split')} num_questions={row.get('num_questions')} top_k={row.get('top_k')}")
        print(f"  embedding_model={row.get('embedding_model')} corpus_name={row.get('corpus_name')} chunking={row.get('chunking')}")
        print(
            f"  avg_em={row.get('avg_em')} avg_f1={row.get('avg_f1')} "
            f"avg_answer_containment={row.get('avg_answer_containment')}"
        )
        print("-" * 80)

    maybe_export_csv(args.export_csv, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified experiment analysis tool for the CS505 RAG project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("summary", help="Show overall experiment summary: chunk stats, coverage, hit@k, final metrics.")
    p.add_argument("--result_path", required=True)
    p.add_argument("--corpus_path")
    p.add_argument("--chunks_path")
    p.add_argument("--qa_dataset", choices=["nq_open", "squad"], default="nq_open")
    p.add_argument("--qa_split", default="validation")
    p.add_argument("--qa_path", default=None)
    p.add_argument("--num_questions", type=int, default=20)
    p.add_argument("--export_csv")
    p.set_defaults(func=command_summary)

    p = subparsers.add_parser("inspect", help="Inspect per-example predictions.")
    p.add_argument("--result_path", required=True)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--sort_by", default="index", choices=["index", "f1", "em", "containment"])
    p.add_argument("--descending", action="store_true")
    p.add_argument("--show_retrieved", action="store_true")
    p.add_argument("--show_chunk_text", action="store_true")
    p.add_argument("--only_errors", action="store_true")
    p.add_argument("--only_nonzero_f1", action="store_true")
    p.add_argument("--export_csv")
    p.set_defaults(func=command_inspect)

    p = subparsers.add_parser("retrieval", help="Analyze retrieval hits and misses.")
    p.add_argument("--result_path", required=True)
    p.add_argument("--show_hits", action="store_true")
    p.add_argument("--show_misses", action="store_true")
    p.add_argument("--show_chunk_text", action="store_true")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--export_csv")
    p.set_defaults(func=command_retrieval)

    p = subparsers.add_parser("diagnose", help="Diagnose whether errors come from corpus, retrieval, or generation.")
    p.add_argument("--corpus_path", required=True)
    p.add_argument("--result_path", required=True)
    p.add_argument("--qa_dataset", choices=["nq_open", "squad"], default="nq_open")
    p.add_argument("--qa_split", default="validation")
    p.add_argument("--qa_path", default=None)
    p.add_argument("--num_questions", type=int, default=20)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--export_csv")
    p.set_defaults(func=command_diagnose)

    p = subparsers.add_parser("compare", help="Compare multiple result files side by side.")
    p.add_argument("--result_paths", nargs="+", required=True)
    p.add_argument("--sort_by", default="avg_f1", choices=["avg_f1", "avg_em", "avg_answer_containment", "top_k", "num_questions"])
    p.add_argument("--descending", action="store_true")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--export_csv")
    p.set_defaults(func=command_compare)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
