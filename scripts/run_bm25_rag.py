import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.retrieval.bm25_retriever import BM25Retriever
from src.generation.generator import SimpleGenerator
from src.eval.qa_metrics import exact_match, f1_score


def main():
    # Example toy corpus
    documents = [
        "Pride and Prejudice was written by Jane Austen.",
        "The capital of France is Paris.",
        "The largest planet in the solar system is Jupiter.",
        "Python is a popular programming language."
    ]

    # Example question and gold answer
    question = "Who wrote Pride and Prejudice?"
    gold_answer = "Jane Austen"

    # Build retriever
    retriever = BM25Retriever(documents)

    # Retrieve top documents
    retrieved_docs = retriever.retrieve(question, top_k=2)

    # Load generator
    generator = SimpleGenerator(model_name="google/flan-t5-base")

    # Generate prediction with context
    prediction = generator.answer_with_context(question, retrieved_docs)

    # Compute metrics
    em = exact_match(prediction, gold_answer)
    f1 = f1_score(prediction, gold_answer)

    # Print results
    print("Question:", question)
    print("Gold Answer:", gold_answer)
    print("\nRetrieved Documents:")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"{i}. {doc}")

    print("\nPrediction:", prediction)
    print("Exact Match:", em)
    print("F1 Score:", round(f1, 4))


if __name__ == "__main__":
    main()