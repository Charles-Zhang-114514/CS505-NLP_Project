from datasets import load_dataset


def load_doc_sample(sample_size: int = 50):
    # Load a simple text dataset
    dataset = load_dataset("ag_news", split=f"train[:{sample_size}]")

    documents = []

    for i, example in enumerate(dataset):
        text = example["text"]

        documents.append({
            "doc_id": f"doc_{i}",
            "text": text
        })

    return documents


def main():
    docs = load_doc_sample(sample_size=20)

    print("Number of documents:", len(docs))

    for i, doc in enumerate(docs[:3], start=1):
        print(f"\nDoc {i}")
        print(doc["text"])


if __name__ == "__main__":
    main()

