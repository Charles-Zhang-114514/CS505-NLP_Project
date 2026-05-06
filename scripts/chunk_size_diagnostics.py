import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def contains_any_answer(text: str, answers: list[str]) -> str | None:
    text_norm = normalize(text)
    for answer in answers:
        answer_norm = normalize(answer)
        if not answer_norm:
            continue
        if answer_norm.isdigit():
            pattern = rf"(?<!\d){re.escape(answer_norm)}(?!\d)"
        else:
            pattern = rf"\b{re.escape(answer_norm)}\b"
        if re.search(pattern, text_norm):
            return answer
    return None


def find_hit_rank(item: dict[str, Any], max_k: int | None = None) -> tuple[int | None, str | None]:
    chunks = item.get("retrieved_chunks", [])
    if max_k is not None:
        chunks = chunks[:max_k]
    for rank, chunk in enumerate(chunks, start=1):
        matched = contains_any_answer(chunk.get("text", ""), item.get("gold_answers", []))
        if matched is not None:
            return rank, matched
    return None, None


def default_label(path: str) -> str:
    name = Path(path).stem
    match = re.search(r"fixed(\d+)", name)
    if match:
        return f"fixed-{match.group(1)}"
    match = re.search(r"semantic", name)
    if match:
        return "semantic"
    return name


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        return plt, np
    except ImportError:
        return None, None


def command_retrieval_heatmap(args: argparse.Namespace) -> None:
    labels = args.labels or [default_label(path) for path in args.result_paths]
    if len(labels) != len(args.result_paths):
        raise ValueError("--labels must have the same length as --result_paths")

    datasets = [(label, load_json(path)) for label, path in zip(labels, args.result_paths)]
    max_examples = min(len(data.get("results", [])) for _, data in datasets)
    if args.num_questions is not None:
        max_examples = min(max_examples, args.num_questions)

    rows = []
    matrix = []
    for idx in range(max_examples):
        matrix_row = []
        for label, data in datasets:
            item = data["results"][idx]
            hit_rank, matched = find_hit_rank(item, args.max_k)
            matrix_row.append(hit_rank or 0)
            rows.append(
                {
                    "index": item.get("index", idx + 1),
                    "question": item.get("query"),
                    "label": label,
                    "hit_rank": hit_rank or "",
                    "hit": int(hit_rank is not None),
                    "matched_answer": matched or "",
                    "top_k": data.get("top_k"),
                    "em": item.get("exact_match"),
                    "f1": item.get("f1_score"),
                    "containment": item.get("answer_containment"),
                }
            )
        matrix.append(matrix_row)

    write_csv(args.output_csv, rows)
    print(f"Saved retrieval CSV to {args.output_csv}")

    if args.output_png:
        plt, np = maybe_import_matplotlib()
        if plt is None:
            print("matplotlib is not installed; skipped PNG output.")
            return

        arr = np.array(matrix, dtype=float)
        fig_height = max(4, min(20, 0.18 * max_examples))
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(labels)), fig_height))
        masked = np.ma.masked_where(arr == 0, arr)
        im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap="viridis_r")
        ax.imshow(arr == 0, aspect="auto", interpolation="nearest", cmap="Greys", alpha=0.15)
        ax.set_title("Retrieval answer-hit rank by chunk setting")
        ax.set_xlabel("Chunk setting")
        ax.set_ylabel("Question index")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        y_ticks = list(range(0, max_examples, max(1, max_examples // 10)))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(i + 1) for i in y_ticks])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Hit rank, lower is better; blank/gray = miss")
        fig.tight_layout()
        ensure_dir(os.path.dirname(args.output_png) or ".")
        fig.savefig(args.output_png, dpi=200)
        plt.close(fig)
        print(f"Saved retrieval heatmap to {args.output_png}")


def build_rag_prompt(question: str, retrieved_chunks: list[dict[str, Any]]) -> tuple[str, str]:
    ordered = sorted(retrieved_chunks, key=lambda x: x.get("rank", 9999))
    context_docs = [chunk.get("text", "") for chunk in ordered]
    context = "\n".join(context_docs)
    prompt = f"Given the following context:\n{context}\n\nAnswer the question: {question}"
    return prompt, context


def find_answer_char_span(prompt: str, answers: list[str]) -> tuple[int | None, int | None, str | None]:
    prompt_lower = prompt.lower()
    for answer in sorted(answers, key=len, reverse=True):
        answer_clean = str(answer).strip()
        if not answer_clean:
            continue
        start = prompt_lower.find(answer_clean.lower())
        if start >= 0:
            return start, start + len(answer_clean), answer_clean
    return None, None, None


def token_answer_mask(offsets: list[tuple[int, int]], start: int | None, end: int | None) -> list[int]:
    if start is None or end is None:
        return [0 for _ in offsets]
    mask = []
    for tok_start, tok_end in offsets:
        overlaps = tok_end > start and tok_start < end
        mask.append(int(overlaps))
    return mask


def safe_token(token: str) -> str:
    return token.replace("\n", "\\n").replace("\t", "\\t")


def command_cross_attention(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    result = load_json(args.result_path)
    items = result["results"]
    if args.example_indices:
        wanted = {int(x) for x in args.example_indices}
        selected = [item for item in items if int(item.get("index", -1)) in wanted]
    else:
        selected = items[: args.max_examples]

    ensure_dir(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.generator_model, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.generator_model)
    model.eval()

    summary_rows = []
    for item in selected:
        question = item["query"]
        prompt, _ = build_rag_prompt(question, item.get("retrieved_chunks", []))
        label_text = item.get("prediction") if args.label_source == "prediction" else item.get("gold_answers", [""])[0]
        if not label_text:
            label_text = item.get("gold_answers", [""])[0]

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        labels = tokenizer(label_text, return_tensors="pt", truncation=True, max_length=args.max_label_tokens).input_ids

        with torch.no_grad():
            output = model(
                **encoded,
                labels=labels,
                output_attentions=True,
                return_dict=True,
            )

        if output.cross_attentions is None:
            raise RuntimeError("Model did not return cross-attentions.")

        layer_tensors = torch.stack([attn.detach().cpu() for attn in output.cross_attentions], dim=0)
        # Shape: layers, batch, heads, decoder_tokens, encoder_tokens.
        token_attention = layer_tensors.mean(dim=(0, 1, 2, 3)).numpy()
        input_ids = encoded["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        ans_start, ans_end, matched_answer = find_answer_char_span(prompt, item.get("gold_answers", []))
        mask = token_answer_mask(offsets, ans_start, ans_end)
        answer_mass = float(sum(float(score) for score, is_answer in zip(token_attention, mask) if is_answer))

        prefix = f"example_{int(item.get('index', 0)):03d}"
        token_csv = os.path.join(args.output_dir, f"{prefix}_cross_attention_tokens.csv")
        token_rows = []
        for i, (tok, score, (char_start, char_end), is_answer) in enumerate(zip(tokens, token_attention, offsets, mask)):
            token_rows.append(
                {
                    "token_index": i,
                    "token": safe_token(tok),
                    "char_start": char_start,
                    "char_end": char_end,
                    "attention": float(score),
                    "is_gold_answer": is_answer,
                }
            )
        write_csv(token_csv, token_rows)

        summary_rows.append(
            {
                "index": item.get("index"),
                "question": question,
                "label_source": args.label_source,
                "label_text": label_text,
                "matched_gold_answer": matched_answer or "",
                "answer_attention_mass": answer_mass,
                "em": item.get("exact_match"),
                "f1": item.get("f1_score"),
                "containment": item.get("answer_containment"),
                "token_csv": token_csv,
            }
        )

        if args.output_png:
            plt, np = maybe_import_matplotlib()
            if plt is not None:
                plot_count = min(args.max_plot_tokens, len(tokens))
                top_scores = token_attention[:plot_count]
                colors = ["tab:red" if x else "tab:blue" for x in mask[:plot_count]]
                fig, ax = plt.subplots(figsize=(max(10, plot_count * 0.18), 4))
                ax.bar(range(plot_count), top_scores, color=colors, width=0.9)
                tick_step = max(1, plot_count // 40)
                ax.set_xticks(range(0, plot_count, tick_step))
                ax.set_xticklabels([safe_token(tokens[i]) for i in range(0, plot_count, tick_step)], rotation=90, fontsize=7)
                ax.set_ylabel("Mean cross-attention")
                ax.set_title(f"Cross-attention over input tokens: example {item.get('index')}")
                fig.tight_layout()
                png_path = os.path.join(args.output_dir, f"{prefix}_cross_attention.png")
                fig.savefig(png_path, dpi=200)
                plt.close(fig)
                summary_rows[-1]["png"] = png_path

        print(
            f"Example {item.get('index')}: answer_attention_mass={answer_mass:.6f}, "
            f"matched_answer={matched_answer}"
        )

    summary_csv = os.path.join(args.output_dir, "cross_attention_summary.csv")
    write_csv(summary_csv, summary_rows)
    print(f"Saved cross-attention summary to {summary_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk-size diagnostics for RAG experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("retrieval-heatmap", help="Plot answer-containing retrieval rank across result files.")
    p.add_argument("--result_paths", nargs="+", required=True)
    p.add_argument("--labels", nargs="+")
    p.add_argument("--num_questions", type=int)
    p.add_argument("--max_k", type=int, default=5)
    p.add_argument("--output_csv", default="results/retrieval_heatmap.csv")
    p.add_argument("--output_png", default="results/retrieval_heatmap.png")
    p.set_defaults(func=command_retrieval_heatmap)

    p = subparsers.add_parser("cross-attention", help="Extract FLAN-T5 cross-attention over retrieved context tokens.")
    p.add_argument("--result_path", required=True)
    p.add_argument("--example_indices", nargs="+")
    p.add_argument("--max_examples", type=int, default=5)
    p.add_argument("--generator_model", default="google/flan-t5-base")
    p.add_argument("--label_source", choices=["prediction", "gold"], default="prediction")
    p.add_argument("--max_input_tokens", type=int, default=512)
    p.add_argument("--max_label_tokens", type=int, default=32)
    p.add_argument("--max_plot_tokens", type=int, default=160)
    p.add_argument("--output_dir", default="results/attention_diagnostics")
    p.add_argument("--output_png", action="store_true")
    p.set_defaults(func=command_cross_attention)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
