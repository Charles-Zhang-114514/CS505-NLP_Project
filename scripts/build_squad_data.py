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

from src.data_prep.squad_loader import load_squad_corpus, load_squad_qa_examples, save_json


def sidecar_metadata_path(path: str) -> str:
    base, _ = os.path.splitext(path)
    return f"{base}.meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SQuAD QA pairs and a deduplicated SQuAD corpus.")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--sample_size", type=int, default=None, help="Optional cap on number of SQuAD examples to load.")
    parser.add_argument("--output_qa_path", required=True)
    parser.add_argument("--output_corpus_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_started_at = datetime.now(timezone.utc).isoformat()
    build_start = time.perf_counter()
    qa_examples = load_squad_qa_examples(split=args.split, sample_size=args.sample_size)
    corpus = load_squad_corpus(split=args.split, sample_size=args.sample_size)

    save_json(qa_examples, args.output_qa_path)
    save_json(corpus, args.output_corpus_path)
    meta_path = sidecar_metadata_path(args.output_corpus_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "script": "build_squad_data.py",
                "split": args.split,
                "sample_size": args.sample_size,
                "output_qa_path": args.output_qa_path,
                "output_corpus_path": args.output_corpus_path,
                "num_qa_examples": len(qa_examples),
                "num_corpus_docs": len(corpus),
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

    print(f"Saved {len(qa_examples)} QA examples to {args.output_qa_path}")
    print(f"Saved {len(corpus)} corpus docs to {args.output_corpus_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
