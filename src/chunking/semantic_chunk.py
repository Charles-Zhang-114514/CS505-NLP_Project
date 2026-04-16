from typing import List, Dict
import math

import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


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
        "method": "semantic",
    }


def _simple_token_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def _cosine_similarity(vec_a, vec_b) -> float:
    dot = vec_a.multiply(vec_b).sum()
    norm_a = math.sqrt(vec_a.multiply(vec_a).sum())
    norm_b = math.sqrt(vec_b.multiply(vec_b).sum())

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def semantic_chunk_document(
    doc_id: str,
    title: str,
    text: str,
    chunk_size: int = 250,
    threshold: float = 0.75,
) -> List[Dict]:
    if not text or not text.strip():
        return []

    # 第一次用 nltk 可能需要这个资源
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        nltk.download("punkt")
        sentences = sent_tokenize(text)

    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    # 如果只有一句话，直接作为一个 chunk
    if len(sentences) == 1:
        return [
            _make_chunk_record(
                doc_id=doc_id,
                title=title,
                chunk_text=sentences[0],
                chunk_index=0,
            )
        ]

    sentence_token_counts = [_simple_token_count(s) for s in sentences]

    vectorizer = TfidfVectorizer()
    sentence_vecs = vectorizer.fit_transform(sentences)

    chunks = []
    current_sentences = [sentences[0]]
    current_token_count = sentence_token_counts[0]
    chunk_index = 0

    for i in range(1, len(sentences)):
        sim = _cosine_similarity(sentence_vecs[i - 1], sentence_vecs[i])
        next_len = sentence_token_counts[i]

        same_chunk = (
            sim >= threshold
            and current_token_count + next_len <= chunk_size
        )

        if same_chunk:
            current_sentences.append(sentences[i])
            current_token_count += next_len
        else:
            chunks.append(
                _make_chunk_record(
                    doc_id=doc_id,
                    title=title,
                    chunk_text=" ".join(current_sentences).strip(),
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
            current_sentences = [sentences[i]]
            current_token_count = next_len

    if current_sentences:
        chunks.append(
            _make_chunk_record(
                doc_id=doc_id,
                title=title,
                chunk_text=" ".join(current_sentences).strip(),
                chunk_index=chunk_index,
            )
        )

    return chunks


def semantic_chunk_documents(
    documents: List[Dict],
    chunk_size: int = 250,
    threshold: float = 0.75,
) -> List[Dict]:
    all_chunks = []

    for doc in documents:
        doc_id = doc["doc_id"]
        title = doc.get("title", "")
        text = doc["text"]

        doc_chunks = semantic_chunk_document(
            doc_id=doc_id,
            title=title,
            text=text,
            chunk_size=chunk_size,
            threshold=threshold,
        )
        all_chunks.extend(doc_chunks)

    return all_chunks