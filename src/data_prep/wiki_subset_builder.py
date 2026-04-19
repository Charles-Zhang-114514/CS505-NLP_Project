from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any

import wikipedia
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize

from src.retrieval.embedder import Embedder


def save_json(data: list[dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _safe_summary(title: str, sentences_per_page: int) -> str | None:
    try:
        return wikipedia.summary(title, sentences=sentences_per_page, auto_suggest=False)
    except Exception:
        return None


def _safe_page_content(title: str) -> str | None:
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return page.content
    except Exception:
        return None


def _ensure_punkt() -> None:
    try:
        sent_tokenize("Test sentence. Another one.")
    except LookupError:
        nltk.download("punkt")


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _make_sentence_windows(sentences: list[str], window_size: int = 3, stride: int = 2) -> list[str]:
    windows: list[str] = []
    if not sentences:
        return windows
    if len(sentences) <= window_size:
        joined = _clean_text(" ".join(sentences))
        return [joined] if joined else []

    for start in range(0, len(sentences), stride):
        window = sentences[start:start + window_size]
        if not window:
            continue
        joined = _clean_text(" ".join(window))
        if joined:
            windows.append(joined)
        if start + window_size >= len(sentences):
            break
    return windows


def build_random_wikipedia_subset(
    num_docs: int,
    sentences_per_page: int = 8,
    seed: int = 42,
    sleep_seconds: float = 0.2,
) -> list[dict[str, Any]]:
    random.seed(seed)
    docs: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    attempts = 0
    max_attempts = max(num_docs * 10, 50)

    while len(docs) < num_docs and attempts < max_attempts:
        attempts += 1
        try:
            title = wikipedia.random(pages=1)
            if isinstance(title, list):
                title = title[0]
        except Exception:
            continue

        if not title or title in seen_titles:
            continue

        text = _safe_summary(title, sentences_per_page)
        if not text:
            continue

        seen_titles.add(title)
        docs.append(
            {
                "doc_id": f"wiki_random_{len(docs)}",
                "title": title,
                "text": text,
                "source": "wikipedia_random",
            }
        )
        print(f"[{len(docs)}/{num_docs}] random title={title!r} ✓")
        time.sleep(sleep_seconds)

    print(f"Built random Wikipedia subset with {len(docs)} docs.")
    return docs


def build_question_conditioned_wikipedia_subset(
    qa_split: str = "validation",
    num_questions: int = 20,
    search_top_n: int = 5,
    pages_per_question: int = 2,
    sentences_per_page: int = 8,
    sleep_seconds: float = 0.2,
) -> list[dict[str, Any]]:
    nq = load_dataset("nq_open", split=f"{qa_split}[:{num_questions}]")

    docs: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for i, example in enumerate(nq):
        question = example["question"]
        try:
            hits = wikipedia.search(question, results=search_top_n)
        except Exception as e:
            print(f"[{i}] search failed for question={question!r}: {e}")
            continue

        kept_this_question = 0
        for title in hits:
            if title in seen_titles:
                continue
            text = _safe_summary(title, sentences_per_page)
            if not text:
                continue

            seen_titles.add(title)
            docs.append(
                {
                    "doc_id": f"wiki_q_{i}_{kept_this_question}",
                    "title": title,
                    "text": text,
                    "source": "wikipedia_question_search",
                    "question_seed": question,
                }
            )
            kept_this_question += 1
            print(f"[q{i+1}/{num_questions}] question={question[:50]!r} -> {title!r} ✓")
            time.sleep(sleep_seconds)

            if kept_this_question >= pages_per_question:
                break

    print(f"Built question-conditioned Wikipedia subset with {len(docs)} docs.")
    return docs


def build_question_conditioned_passage_subset(
    qa_split: str = "validation",
    num_questions: int = 20,
    search_top_n: int = 5,
    pages_per_question: int = 2,
    passages_per_page: int = 1,
    window_size: int = 3,
    stride: int = 2,
    sleep_seconds: float = 0.2,
    embed_model_name: str = "BAAI/bge-small-en-v1.5",
) -> list[dict[str, Any]]:
    """
    Recommended next step for question-conditioned corpus construction.

    For each query:
      1) search related Wikipedia pages using the query only
      2) fetch full page content
      3) split into sentence windows
      4) score each window against the query with dense similarity
      5) keep the top passage(s) from each page

    This keeps more useful evidence than summary-only extraction while avoiding
    gold-answer leakage.
    """
    _ensure_punkt()
    nq = load_dataset("nq_open", split=f"{qa_split}[:{num_questions}]")
    embedder = Embedder(model_name=embed_model_name)

    docs: list[dict[str, Any]] = []
    seen_doc_keys: set[str] = set()

    for i, example in enumerate(nq):
        question = example["question"]
        try:
            hits = wikipedia.search(question, results=search_top_n)
        except Exception as e:
            print(f"[{i}] search failed for question={question!r}: {e}")
            continue

        kept_pages = 0
        query_vec = embedder.encode_query(question)

        for title in hits:
            if kept_pages >= pages_per_question:
                break

            content = _safe_page_content(title)
            if not content:
                continue

            content = _clean_text(content)
            if not content:
                continue

            sentences = [s.strip() for s in sent_tokenize(content) if s.strip()]
            windows = _make_sentence_windows(sentences, window_size=window_size, stride=stride)
            if not windows:
                continue

            window_vecs = embedder.encode_documents(windows)
            scores = window_vecs @ query_vec
            ranked_idx = scores.argsort()[::-1][:passages_per_page]

            kept_any = False
            for local_rank, idx in enumerate(ranked_idx):
                passage_text = windows[int(idx)]
                doc_key = f"{title}__{int(idx)}"
                if doc_key in seen_doc_keys:
                    continue
                seen_doc_keys.add(doc_key)
                kept_any = True

                docs.append(
                    {
                        "doc_id": f"wiki_passage_q{i}_{kept_pages}_{local_rank}",
                        "title": title,
                        "text": passage_text,
                        "source": "wikipedia_question_passage_search",
                        "question_seed": question,
                        "passage_score": float(scores[int(idx)]),
                    }
                )

            if kept_any:
                kept_pages += 1
                print(
                    f"[q{i+1}/{num_questions}] question={question[:50]!r} -> {title!r} "
                    f"with {passages_per_page} passage(s) ✓"
                )
                time.sleep(sleep_seconds)

    print(f"Built question-conditioned passage subset with {len(docs)} docs.")
    return docs
