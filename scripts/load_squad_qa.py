from datasets import load_dataset


def load_squad_qa_sample(split: str = "train", sample_size: int = 20, seed: int = 42):
    """
    take question and answers from SQuAD.
    return same format as  load_nq_open_sample:
    [
        {
            "question": ...,
            "answers": [...]
        },
        ...
    ]
    """
    dataset = load_dataset("squad", split=split)
    dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    examples = []
    for ex in dataset:
        answers = ex["answers"]["text"] if "answers" in ex and "text" in ex["answers"] else []

        examples.append({
            "question": ex["question"],
            "answers": answers,
        })

    return examples


def main():
    examples = load_squad_qa_sample(sample_size=5)

    print("Loaded", len(examples), "examples")
    for i, ex in enumerate(examples, start=1):
        print(f"\nExample {i}")
        print("Question:", ex["question"])
        print("Answers:", ex["answers"])


if __name__ == "__main__":
    main()