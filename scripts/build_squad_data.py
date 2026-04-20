import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data_prep.squad_loader import load_squad_corpus, load_squad_qa_examples, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SQuAD QA pairs and a deduplicated SQuAD corpus.")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--sample_size", type=int, default=None, help="Optional cap on number of SQuAD examples to load.")
    parser.add_argument("--output_qa_path", required=True)
    parser.add_argument("--output_corpus_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qa_examples = load_squad_qa_examples(split=args.split, sample_size=args.sample_size)
    corpus = load_squad_corpus(split=args.split, sample_size=args.sample_size)

    save_json(qa_examples, args.output_qa_path)
    save_json(corpus, args.output_corpus_path)

    print(f"Saved {len(qa_examples)} QA examples to {args.output_qa_path}")
    print(f"Saved {len(corpus)} corpus docs to {args.output_corpus_path}")


if __name__ == "__main__":
    main()
