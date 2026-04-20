from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from datasets import load_dataset


def _stable_doc_id(title: str, context: str) -> str:
    key = f"{title}||{context}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()[:12]
    safe_title = title.replace(" ", "_") if title else "untitled"
    return f"squad_{safe_title}_{digest}"


def _normalize_answers(answers: Any) -> list[str]:
    if isinstance(answers, dict):
        texts = answers.get("text", [])
        return [str(t).strip() for t in texts if str(t).strip()]
    if isinstance(answers, list):
        return [str(t).strip() for t in answers if str(t).strip()]
    return []


def load_squad_qa_examples(split: str = "validation", sample_size: int | None = None) -> list[dict[str, Any]]:
    dataset = load_dataset("squad", split=split)
    if sample_size is not None:
        dataset = dataset.select(range(min(sample_size, len(dataset))))

    examples: list[dict[str, Any]] = []
    for ex in dataset:
        title = ex.get("title", "")
        context = ex["context"].strip()
        answers = _normalize_answers(ex.get("answers", {}))
        if not answers:
            continue
        examples.append(
            {
                "id": ex.get("id"),
                "question": ex["question"].strip(),
                "answers": answers,
                "context": context,
                "title": title,
                "doc_id": _stable_doc_id(title, context),
            }
        )
    return examples


def load_squad_corpus(split: str = "validation", sample_size: int | None = None) -> list[dict[str, Any]]:
    dataset = load_dataset("squad", split=split)
    if sample_size is not None:
        dataset = dataset.select(range(min(sample_size, len(dataset))))

    docs: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    for ex in dataset:
        title = ex.get("title", "")
        context = ex["context"].strip()
        if not context:
            continue
        doc_id = _stable_doc_id(title, context)
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        docs.append(
            {
                "doc_id": doc_id,
                "title": title,
                "text": context,
                "source": "squad_context",
            }
        )
    return docs


def save_json(data: list[dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
