import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.generation.generator import SimpleGenerator
from src.eval.qa_metrics import exact_match, f1_score


def main():
    # Example question and gold answer
    question = "Who wrote Pride and Prejudice?"
    gold_answer = "Jane Austen"

    # Load generator
    generator = SimpleGenerator(model_name="google/flan-t5-base")

    # Generate prediction
    prediction = generator.answer_question(question)

    # Compute metrics
    em = exact_match(prediction, gold_answer)
    f1 = f1_score(prediction, gold_answer)

    # Print results
    print("Question:", question)
    print("Gold Answer:", gold_answer)
    print("Prediction:", prediction)
    print("Exact Match:", em)
    print("F1 Score:", round(f1, 4))


if __name__ == "__main__":
    main()