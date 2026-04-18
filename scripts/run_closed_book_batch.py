import json
import os
import sys
from datasets import load_dataset

# Add src folder to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from generation.generator import SimpleGenerator
from eval.qa_metrics import exact_match, f1_score


def load_nq_open_sample(split: str = "train", sample_size: int = 5):
    # Load a small sample from NQ Open
    dataset = load_dataset("nq_open", split=f"{split}[:{sample_size}]")

    # Convert to simple format
    examples = []
    for example in dataset:
        examples.append({
            "question": example["question"],
            "answers": example["answer"]
        })

    return examples


def max_em_over_answers(prediction: str, gold_answers: list[str]) -> float:
    # Take the best EM among all gold answers
    scores = [exact_match(prediction, ans) for ans in gold_answers]
    return max(scores)


def max_f1_over_answers(prediction: str, gold_answers: list[str]) -> float:
    # Take the best F1 among all gold answers
    scores = [f1_score(prediction, ans) for ans in gold_answers]
    return max(scores)


def main():
    # Load sample data
    examples = load_nq_open_sample(split="validation", sample_size=20)

    # Load generator
    generator = SimpleGenerator(model_name="google/flan-t5-base")

    total_em = 0.0
    total_f1 = 0.0

    # Assign results
    all_results = []
    
    # Run batch evaluation
    for i, example in enumerate(examples, start=1):
        question = example["question"]
        gold_answers = example["answers"]

        prediction = generator.answer_question(question)

        em = max_em_over_answers(prediction, gold_answers)
        f1 = max_f1_over_answers(prediction, gold_answers)

        total_em += em
        total_f1 += f1
        
        all_results.append({
            "question": question,
            "gold_answers": gold_answers,
            "prediction": prediction,
            "em": em,
            "f1": f1
        })

        print(f"\nExample {i}")
        print("Question:", question)
        print("Gold Answers:", gold_answers)
        print("Prediction:", prediction)
        print("EM:", em)
        print("F1:", round(f1, 4))

    avg_em = total_em / len(examples)
    avg_f1 = total_f1 / len(examples)

    print("\n===== Final Results =====")
    print("Number of examples:", len(examples))
    print("Average EM:", round(avg_em, 4))
    print("Average F1:", round(avg_f1, 4))
    
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "closed_book_val20_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()