import re
from typing import List, Dict


def _simple_tokenize(text: str) -> List[str]:
    if not text:
        return []
    return text.split()


def _make_chunk_record(
    doc_id: str,
    title: str,
    chunk_text: str,
    chunk_index: int,
) -> Dict:
    return {
        "chunk_id": f"{doc_id}_chunk_{chunk_index}",
        "doc_id": doc_id,
        "title": title,
        "text": chunk_text,
        "chunk_index": chunk_index,
        "method": "fixed",
    }


def fixed_chunk_document(
    doc_id: str,
    title: str,
    text: str,
    chunk_size: int = 120,
    overlap: int = 20,
) -> List[Dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = _simple_tokenize(text)
    if not tokens:
        return []

    chunks = []
    step = chunk_size - overlap
    chunk_index = 0

    for start in range(0, len(tokens), step):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            continue

        chunk_text = " ".join(chunk_tokens).strip()
        chunks.append(
            _make_chunk_record(
                doc_id=doc_id,
                title=title,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
            )
        )
        chunk_index += 1

        if end >= len(tokens):
            break

    return chunks


def fixed_chunk_documents(
    documents: List[Dict],
    chunk_size: int = 120,
    overlap: int = 20,
) -> List[Dict]:
    all_chunks = []

    for doc in documents:
        doc_id = doc["doc_id"]
        title = doc.get("title", "")
        text = doc["text"]

        doc_chunks = fixed_chunk_document(
            doc_id=doc_id,
            title=title,
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(doc_chunks)

    return all_chunks