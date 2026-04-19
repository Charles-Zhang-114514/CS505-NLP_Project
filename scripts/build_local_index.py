import argparse
import json
import os
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.retrieval.embedder import Embedder


def load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local dense index from chunked corpus.")
    parser.add_argument("--input_chunks", required=True)
    parser.add_argument("--embedding_model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = load_json(args.input_chunks)
    texts = [c["text"] for c in chunks]

    embedder = Embedder(model_name=args.embedding_model)
    embeddings = embedder.encode_documents(texts)

    os.makedirs(args.output_dir, exist_ok=True)
    chunks_out = os.path.join(args.output_dir, "chunks.json")
    emb_out = os.path.join(args.output_dir, "embeddings.npy")
    meta_out = os.path.join(args.output_dir, "metadata.json")

    with open(chunks_out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    np.save(emb_out, embeddings)
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model": args.embedding_model,
                "num_chunks": len(chunks),
                "dimension": int(embeddings.shape[1]) if len(embeddings) else 0,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved chunks to      {chunks_out}")
    print(f"Saved embeddings to {emb_out}")
    print(f"Saved metadata to   {meta_out}")


if __name__ == "__main__":
    main()
