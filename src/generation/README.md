# Stage 4 — Generation

## Overview

This module implements the generation stage of the RAG pipeline using **Qwen3.5-4B**.
It accepts retrieved chunks from Stage 3 and generates grounded answers, evaluated with Exact Match (EM) and token-level F1.

---

## Files

- `generator.py` — QwenGenerator class (main implementation)
- `run.bash` — quick test script
- `README.md` — this file

---

## How to Run

### Environment Setup (SCC)

```bash
module unload python3
module load miniconda
conda activate rag
export HF_HOME=/projectnb/cs505am/students/meiaohan/hf_cache
cd /projectnb/cs505am/students/meiaohan/CS505-NLP_Project
```

### Quick Test

```bash
bash src/generation/run.bash
```

### Full Evaluation (requires Stage 3 output)

```bash
python scripts/run_qwen_rag.py \
    --input results/retrieved.json \
    --output results/generation_output.json \
    --mode rag
```

---

## Input / Output Format

### Input (`retrieved_chunks` from Stage 3)

```json
[
  {
    "chunk_id": "doc1_0",
    "doc_id": "doc1",
    "text": "Pride and Prejudice is a novel by Jane Austen.",
    "score": 0.92,
    "rank": 1
  }
]
```

### Output (`generation_result`)

```json
{
  "answer": "Jane Austen",
  "query": "Who wrote Pride and Prejudice?",
  "context_used": ["Pride and Prejudice is a novel by Jane Austen."],
  "model": "Qwen/Qwen3.5-4B"
}
```

---

## Example Result

| Question | Answer |
|----------|--------|
| Who wrote Pride and Prejudice? | Jane Austen ✅ |

---

## Notes

- Model: `Qwen/Qwen3.5-4B` (9.3GB, float16)
- GPU: NVIDIA L40S on SCC (46GB VRAM)
- Model cache: `/projectnb/cs505am/students/meiaohan/hf_cache`
- `enable_thinking` is disabled by default for faster inference