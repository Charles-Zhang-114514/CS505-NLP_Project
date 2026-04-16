# scripts/load_wiki_oracle.py
# Build an "oracle-ish" Wikipedia corpus for NQ Open evaluation.

import os
import json
import time
from datasets import load_dataset
import wikipedia


def load_wiki_oracle_docs(nq_sample_size: int = 20, sentences_per_doc: int = 10):
    """
    Oracle-ish corpus for NQ Open.
    Strategy: search Wikipedia with gold answer, only keep pages
    whose summary actually contains the gold answer text.
    """
    nq = load_dataset("nq_open", split=f"validation[:{nq_sample_size}]")

    documents = []
    seen_titles = set()
    kept = 0
    dropped = 0

    for i, example in enumerate(nq):
        question = example["question"]
        answers = example["answer"]
        if not answers:
            continue
        gold = answers[0]

        try:
            hits = wikipedia.search(gold, results=5)
        except Exception as e:
            print(f"[{i}] search failed for {gold!r}: {e}")
            dropped += 1
            continue

        if not hits:
            print(f"[{i}] no hit for gold={gold!r}")
            dropped += 1
            continue

        chosen_title = None
        chosen_text = None
        for title in hits:
            if title in seen_titles:
                continue
            try:
                text = wikipedia.summary(
                    title, sentences=sentences_per_doc, auto_suggest=False
                )
            except Exception:
                continue
            if gold.lower() in text.lower():
                chosen_title = title
                chosen_text = text
                break

        if chosen_title is None:
            print(f"[{i}] no page contains gold={gold!r}, DROPPED")
            dropped += 1
            continue

        seen_titles.add(chosen_title)
        documents.append({
            "doc_id": f"wiki_{i}",
            "title": chosen_title,
            "text": chosen_text,
        })
        kept += 1
        print(f"[{i}] Q={question[:40]!r}  gold={gold!r}  -> {chosen_title!r} ✓")

        time.sleep(0.3)

    print(f"\nBuilt oracle corpus: kept {kept}, dropped {dropped}")
    return documents


if __name__ == "__main__":
    docs = load_wiki_oracle_docs(nq_sample_size=20)

    # Preview first 3
    print("\n=== First 3 docs preview ===")
    for d in docs[:3]:
        print("---")
        print(d["doc_id"], d["title"])
        print(d["text"][:300])

    # Save to JSON
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(PROJECT_ROOT, "data", "corpus")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wiki_oracle.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(docs)} docs to {out_path}")