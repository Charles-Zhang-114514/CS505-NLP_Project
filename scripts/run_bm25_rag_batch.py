import os
import sys
from datasets import load_dataset

# Add src folder to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from retrieval.bm25_retriever import BM25Retriever
from generation.generator import SimpleGenerator
from eval.qa_metrics import exact_match, f1_score


def load_nq_open_sample(split: str = "train", sample_size: int = 5):
    # Load a small sample from NQ Open
    dataset = load_dataset("nq_open", split=f"{split}[:{sample_size}]")

    examples = []
    for example in dataset:
        examples.append({
            "question": example["question"],
            "answers": example["answer"]
        })

    return examples


def load_doc_sample(sample_size: int = 50):
    # Load a simple text dataset as document database
    dataset = load_dataset("ag_news", split=f"train[:{sample_size}]")

    documents = []
    for i, example in enumerate(dataset):
        documents.append({
            "doc_id": f"doc_{i}",
            "text": example["text"]
        })

    return documents


def max_em_over_answers(prediction: str, gold_answers: list[str]) -> float:
    return max([exact_match(prediction, ans) for ans in gold_answers])


def max_f1_over_answers(prediction: str, gold_answers: list[str]) -> float:
    return max([f1_score(prediction, ans) for ans in gold_answers])


def main():
    # Load QA examples
    examples = load_nq_open_sample(sample_size=5)

    # Load document database
    doc_objects = load_doc_sample(sample_size=20)
    documents = [doc["text"] for doc in doc_objects]

    # Build retriever
    retriever = BM25Retriever(documents)

    # Load generator
    generator = SimpleGenerator(model_name="google/flan-t5-base")

    total_em = 0.0
    total_f1 = 0.0

    for i, example in enumerate(examples, start=1):
        question = example["question"]
        gold_answers = example["answers"]

        # Retrieve documents
        retrieved_docs = retriever.retrieve(question, top_k=2)

        # Generate answer with context
        prediction = generator.answer_with_context(question, retrieved_docs)

        # Evaluate
        em = max_em_over_answers(prediction, gold_answers)
        f1 = max_f1_over_answers(prediction, gold_answers)

        total_em += em
        total_f1 += f1

        print(f"\nExample {i}")
        print("Question:", question)
        print("Gold Answers:", gold_answers)

        print("\nRetrieved Docs:")
        for doc in retrieved_docs:
            print("-", doc)

        print("\nPrediction:", prediction)
        print("EM:", em)
        print("F1:", round(f1, 4))

    avg_em = total_em / len(examples)
    avg_f1 = total_f1 / len(examples)

    print("\n===== Final Results =====")
    print("Number of examples:", len(examples))
    print("Average EM:", round(avg_em, 4))
    print("Average F1:", round(avg_f1, 4))


if __name__ == "__main__":
    main()