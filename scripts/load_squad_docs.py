from datasets import load_dataset


def load_squad_docs(sample_size: int = 100, seed: int = 42):
    dataset = load_dataset("squad", split="train")
    dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    documents = []
    seen_texts = set()

    for i, example in enumerate(dataset):
        context = example["context"].strip()

        # Skip duplicate contexts
        if context in seen_texts:
            continue
        seen_texts.add(context)

        documents.append({
            "doc_id": f"doc_{i}",
            "text": context
        })

    return documents