import argparse
import json
import os
import sys
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src", "retrieval"))

from src.retrieval.qdrant_indexer import QdrantIndexer


def load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload chunked corpus to a Qdrant collection.")
    parser.add_argument("--input_chunks", required=True)
    parser.add_argument("--collection_name", required=True)
    parser.add_argument("--embedding_model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = load_json(args.input_chunks)
    if not chunks:
        raise ValueError("No chunks found in input file.")

    indexer = QdrantIndexer(
        collection_name=args.collection_name,
        model_name=args.embedding_model,
    )
    indexer.create_collection(recreate=args.recreate)
    indexer.index_chunks(chunks, batch_size=args.batch_size)
    print(f"Collection {args.collection_name!r} now contains {indexer.count_points()} points.")


if __name__ == "__main__":
    main()
