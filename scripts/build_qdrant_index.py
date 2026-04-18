import os
import sys
import argparse
from typing import List, Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if os.path.basename(PROJECT_ROOT) == "scripts":
    PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src", "retrieval"))

from src.chunking.fixed_chunk import fixed_chunk_document
from src.chunking.semantic_chunk import semantic_chunk_document
from src.retrieval.qdrant_indexer import QdrantIndexer


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


def load_corpus(corpus_name: str, sample_size: int) -> List[Dict[str, Any]]:
    if corpus_name == "ag_news":
        from scripts.load_wiki import load_doc_sample
        docs = load_doc_sample(sample_size=sample_size)
    elif corpus_name == "squad":
        from scripts.load_squad_docs import load_squad_docs
        docs = load_squad_docs(sample_size=sample_size)
    elif corpus_name == "wiki_oracle":
        from scripts.load_wiki_oracle import load_wiki_oracle_docs
        docs = load_wiki_oracle_docs(nq_sample_size=sample_size)
    else:
        raise ValueError(f"Unsupported corpus: {corpus_name}")
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
        return [
            {
                "chunk_id": f"{doc['doc_id']}_chunk_0",
                "doc_id": doc["doc_id"],
                "title": doc.get("title", ""),
                "text": doc["text"],
                "chunk_index": 0,
                "method": "raw",
            }
            for doc in documents
        ]

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
            raise ValueError(f"Unsupported chunking: {chunking}")
        all_chunks.extend(doc_chunks)
    return all_chunks


def parse_args():
    parser = argparse.ArgumentParser(description="Build and upload a Qdrant collection for the CS505 project.")
    parser.add_argument("--corpus", choices=["ag_news", "squad", "wiki_oracle"], required=True)
    parser.add_argument("--corpus_sample_size", type=int, default=20)
    parser.add_argument("--chunking", choices=["raw", "fixed", "semantic"], default="raw")
    parser.add_argument("--fixed_chunk_size", type=int, default=120)
    parser.add_argument("--fixed_overlap", type=int, default=20)
    parser.add_argument("--semantic_chunk_size", type=int, default=250)
    parser.add_argument("--semantic_threshold", type=float, default=0.75)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate collection if it already exists")
    return parser.parse_args()


def main():
    args = parse_args()

    docs = load_corpus(args.corpus, args.corpus_sample_size)
    print(f"Loaded {len(docs)} documents from corpus={args.corpus}")

    chunks = build_chunks(
        documents=docs,
        chunking=args.chunking,
        fixed_chunk_size=args.fixed_chunk_size,
        fixed_overlap=args.fixed_overlap,
        semantic_chunk_size=args.semantic_chunk_size,
        semantic_threshold=args.semantic_threshold,
    )
    print(f"Built {len(chunks)} chunks using chunking={args.chunking}")

    if not chunks:
        raise ValueError("No chunks were produced. Cannot build collection.")

    indexer = QdrantIndexer()
    print(f"Target collection: {indexer.collection_name}")
    indexer.create_collection(recreate=args.recreate)
    indexer.index_chunks(chunks, batch_size=args.batch_size)
    count = indexer.count_points()
    print(f"Done. Collection now contains {count} points.")


if __name__ == "__main__":
    main()
