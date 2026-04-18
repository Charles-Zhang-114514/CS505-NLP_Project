from __future__ import annotations

import json
import os
import random
import time
from typing import Any

import wikipedia
from datasets import load_dataset


def save_json(data: list[dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _safe_summary(title: str, sentences_per_page: int) -> str | None:
    try:
        return wikipedia.summary(title, sentences=sentences_per_page, auto_suggest=False)
    except Exception:
        return None


def build_random_wikipedia_subset(
    num_docs: int,
    sentences_per_page: int = 8,
    seed: int = 42,
    sleep_seconds: float = 0.2,
) -> list[dict[str, Any]]:
    """
    Strict-er A-style subset construction.
    Build a fixed Wikipedia subset without using evaluation questions.
    Uses wikipedia.random() to sample pages, then stores summaries locally.
    """
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
    """
    Practical compromise.
    Uses ONLY questions (never gold answers) to search Wikipedia, then unions
    and deduplicates returned pages into a fixed local subset.
    """
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
