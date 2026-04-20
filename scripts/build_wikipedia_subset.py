import argparse
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data_prep.wiki_subset_builder import (
    build_random_wikipedia_subset,
    build_question_conditioned_wikipedia_subset,
    build_question_conditioned_passage_subset,
    save_json,
)


def sidecar_metadata_path(path: str) -> str:
    base, _ = os.path.splitext(path)
    return f"{base}.meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed local Wikipedia subset for RAG.")
    parser.add_argument(
        "--strategy",
        choices=["random", "question", "question_passage"],
        required=True,
        help="random = fixed random pages; question = summary-only query-conditioned pages; question_passage = query-conditioned passage extraction from full page content",
    )
    parser.add_argument("--output_path", required=True)

    # random subset params
    parser.add_argument("--num_docs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    # question-conditioned subset params
    parser.add_argument("--qa_split", default="validation")
    parser.add_argument("--num_questions", type=int, default=20)
    parser.add_argument("--search_top_n", type=int, default=5)
    parser.add_argument("--pages_per_question", type=int, default=2)

    # summary-only params
    parser.add_argument("--sentences_per_page", type=int, default=8)

    # passage-selection params
    parser.add_argument("--passages_per_page", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=3, help="Number of sentences per passage window")
    parser.add_argument("--stride", type=int, default=2, help="Sentence stride between windows")
    parser.add_argument("--embedding_model", default="BAAI/bge-small-en-v1.5")

    # shared
    parser.add_argument("--sleep_seconds", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_started_at = datetime.now(timezone.utc).isoformat()
    build_start = time.perf_counter()

    if args.strategy == "random":
        docs = build_random_wikipedia_subset(
            num_docs=args.num_docs,
            sentences_per_page=args.sentences_per_page,
            seed=args.seed,
            sleep_seconds=args.sleep_seconds,
        )
    elif args.strategy == "question":
        docs = build_question_conditioned_wikipedia_subset(
            qa_split=args.qa_split,
            num_questions=args.num_questions,
            search_top_n=args.search_top_n,
            pages_per_question=args.pages_per_question,
            sentences_per_page=args.sentences_per_page,
            sleep_seconds=args.sleep_seconds,
        )
    else:
        docs = build_question_conditioned_passage_subset(
            qa_split=args.qa_split,
            num_questions=args.num_questions,
            search_top_n=args.search_top_n,
            pages_per_question=args.pages_per_question,
            passages_per_page=args.passages_per_page,
            window_size=args.window_size,
            stride=args.stride,
            sleep_seconds=args.sleep_seconds,
            embed_model_name=args.embedding_model, 
        )

    save_json(docs, args.output_path)
    meta_path = sidecar_metadata_path(args.output_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "script": "build_wikipedia_subset.py",
                "strategy": args.strategy,
                "output_path": args.output_path,
                "num_docs_requested": args.num_docs,
                "num_docs_built": len(docs),
                "qa_split": args.qa_split,
                "num_questions": args.num_questions,
                "search_top_n": args.search_top_n,
                "pages_per_question": args.pages_per_question,
                "sentences_per_page": args.sentences_per_page,
                "passages_per_page": args.passages_per_page,
                "window_size": args.window_size,
                "stride": args.stride,
                "embedding_model": args.embedding_model,
                "seed": args.seed,
                "sleep_seconds": args.sleep_seconds,
                "build_started_at": build_started_at,
                "build_finished_at": datetime.now(timezone.utc).isoformat(),
                "total_runtime_sec": round(time.perf_counter() - build_start, 6),
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved {len(docs)} docs to {args.output_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
