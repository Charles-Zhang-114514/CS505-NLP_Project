import argparse
import json
import os
import sys
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.chunking.fixed_chunk import fixed_chunk_document
from src.chunking.semantic_chunk import semantic_chunk_document


def load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_chunks(
    docs: list[dict[str, Any]],
    chunking: str,
    fixed_chunk_size: int,
    fixed_overlap: int,
    semantic_chunk_size: int,
    semantic_threshold: float,
) -> list[dict[str, Any]]:
    if chunking == "raw":
        return [
            {
                "chunk_id": f"{doc['doc_id']}_chunk_0",
                "doc_id": doc["doc_id"],
                "title": doc.get("title", ""),
                "text": doc["text"],
                "chunk_index": 0,
                "method": "raw",
            }
            for doc in docs
        ]

    chunks: list[dict[str, Any]] = []
    for doc in docs:
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
            raise ValueError(f"Unsupported chunking={chunking}")
        chunks.extend(doc_chunks)
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk a local corpus JSON into retrieval units.")
    parser.add_argument("--input_corpus", required=True)
    parser.add_argument("--chunking", choices=["raw", "fixed", "semantic"], required=True)
    parser.add_argument("--fixed_chunk_size", type=int, default=120)
    parser.add_argument("--fixed_overlap", type=int, default=20)
    parser.add_argument("--semantic_chunk_size", type=int, default=250)
    parser.add_argument("--semantic_threshold", type=float, default=0.75)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs = load_json(args.input_corpus)
    chunks = build_chunks(
        docs=docs,
        chunking=args.chunking,
        fixed_chunk_size=args.fixed_chunk_size,
        fixed_overlap=args.fixed_overlap,
        semantic_chunk_size=args.semantic_chunk_size,
        semantic_threshold=args.semantic_threshold,
    )
    save_json(chunks, args.output_path)
    print(f"Loaded {len(docs)} docs")
    print(f"Built {len(chunks)} chunks with chunking={args.chunking}")
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
