from datasets import load_dataset


def load_nq_open_sample(split: str = "train", sample_size: int = 5):
    # Load a small sample from NQ Open
    dataset = load_dataset("nq_open", split=f"{split}[:{sample_size}]")

    # Convert to a simple list of dicts
    examples = []
    for example in dataset:
        examples.append({
            "question": example["question"],
            "answers": example["answer"]
        })

    return examples


def main():
    # Load sample examples
    examples = load_nq_open_sample(sample_size=5)

    # Print object info
    print("Type of examples:", type(examples))
    print("Type of first item:", type(examples[0]))
    print("Keys of first item:", examples[0].keys())

    # Print examples
    for i, example in enumerate(examples, start=1):
        print(f"\nExample {i}")
        print("Question:", example["question"])
        print("Answers:", example["answers"])


if __name__ == "__main__":
    main()

