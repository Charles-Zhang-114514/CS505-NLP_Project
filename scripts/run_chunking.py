import os
import json
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from load_wiki import load_doc_sample
from src.chunking.fixed_chunk import fixed_chunk_documents
from src.chunking.semantic_chunk import semantic_chunk_documents


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    # 1. load docs
    docs = load_doc_sample(sample_size=20)

    print(f"Loaded {len(docs)} docs")

    # 2. fixed chunk
    fixed_chunks = fixed_chunk_documents(
        docs,
        chunk_size=120,
        overlap=20,
    )

    print(f"Fixed chunking produced {len(fixed_chunks)} chunks")
    if fixed_chunks:
        print("First fixed chunk:")
        print(json.dumps(fixed_chunks[0], indent=2, ensure_ascii=False))

    save_json(fixed_chunks, "data/chunks/fixed_chunks.json")

    # 3. semantic chunk
    semantic_chunks = semantic_chunk_documents(
        docs,
        chunk_size=250,
        threshold=0.75,
    )

    print(f"Semantic chunking produced {len(semantic_chunks)} chunks")
    if semantic_chunks:
        print("First semantic chunk:")
        print(json.dumps(semantic_chunks[0], indent=2, ensure_ascii=False))

    save_json(semantic_chunks, "data/chunks/semantic_chunks.json")


if __name__ == "__main__":
    main()