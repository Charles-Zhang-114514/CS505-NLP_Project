# CS505 NLP Project - Modular RAG Experiments

This repository contains a modular question answering (QA) experimentation framework for Retrieval-Augmented Generation (RAG). It supports:

- closed-book QA (no retrieval),
- sparse retrieval (BM25),
- dense retrieval with local embeddings,
- dense retrieval with Qdrant,
- multiple chunking strategies (raw, fixed, semantic),
- multiple generators (`flan-t5-base` via `SimpleGenerator`, optional Qwen via `QwenGenerator`),
- unified evaluation + experiment analysis.

The current canonical workflow is centered on `scripts/run_experiment.py` and `scripts/analyze_experiment.py`.

## What Is Implemented

### Retrieval modes

- `closed_book`: generator answers without retrieval.
- `bm25`: BM25 over local chunk JSON.
- `local_dense`: cosine similarity over local `.npy` embeddings.
- `qdrant_dense`: vector search in a Qdrant collection.

### QA datasets

- `nq_open` via Hugging Face (`datasets`).
- `squad` via local preprocessing helpers in `src/data_prep/squad_loader.py`.
- custom local QA JSON via `--qa_path`.

### Chunking methods

- `raw`: one chunk per document.
- `fixed`: token window with overlap.
- `semantic`: sentence-level chunking using TF-IDF similarity + thresholding.

### Evaluation metrics

Implemented in `src/eval/qa_metrics.py` and used by experiment runners:

- Exact Match (EM),
- token-level F1,
- answer containment (whether any gold answer string appears in prediction).

## Repository Layout

```text
CS505-NLP_Project/
├── scripts/
│   ├── run_experiment.py          # unified experiment runner
│   ├── analyze_experiment.py      # unified analysis CLI
│   ├── build_squad_data.py        # build QA + corpus JSON from SQuAD
│   ├── build_chunks.py            # raw/fixed/semantic chunking
│   ├── build_local_index.py       # local dense index (.npy embeddings)
│   ├── build_qdrant_index.py      # upload chunks to Qdrant
│   ├── build_wikipedia_subset.py  # random or question-conditioned wiki corpus
│   ├── chunk_size_diagnostics.py  # chunk-size and attention diagnostics
│   └── ...                        # legacy baseline scripts
├── src/
│   ├── generation/generator.py
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── embedder.py
│   │   ├── local_dense_retriever.py
│   │   ├── dense_retriever.py
│   │   └── qdrant_indexer.py
│   ├── chunking/
│   │   ├── fixed_chunk.py
│   │   └── semantic_chunk.py
│   ├── data_prep/
│   │   ├── squad_loader.py
│   │   └── wiki_subset_builder.py
│   └── eval/qa_metrics.py
├── data/
│   ├── qa/       # QA subsets
│   ├── corpus/   # corpus JSON files
│   ├── chunks/   # chunk JSON files
│   └── index/    # local dense index outputs
├── docs/         # report notes and figures
├── results/      # experiment outputs and CSV comparisons
├── requirements.txt
└── README.md
```

## Environment Setup

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Bounds such as `numpy<2` and `aiohttp<3.10` are pinned for resolver compatibility across the stack above.

## Quick Start

Run a small closed-book sanity check first:

```bash
python scripts/run_experiment.py \
  --mode closed_book \
  --qa_dataset squad \
  --qa_split validation \
  --num_questions 20 \
  --generator_type simple \
  --generator_model google/flan-t5-base \
  --output_dir results \
  --output_name quick_closed_book_squad_q20.json
```

Then inspect the summary:

```bash
python scripts/analyze_experiment.py summary \
  --result_path results/quick_closed_book_squad_q20.json \
  --qa_dataset squad \
  --qa_split validation \
  --num_questions 20
```

For Windows PowerShell, replace line-continuation backslashes (`\`) with backticks (`` ` ``), or run each command on one line.

## End-to-End Workflow

The typical pipeline is:

1. build QA/corpus data,
2. build chunks,
3. (optional) build dense index (local or Qdrant),
4. run experiments,
5. analyze result JSON files.

### 1) Build SQuAD QA + corpus JSON

```bash
python scripts/build_squad_data.py \
  --split validation \
  --sample_size 2000 \
  --output_qa_path data/qa/squad_validation_qa_2000.json \
  --output_corpus_path data/corpus/squad_validation_corpus_2000.json
```

This creates:

- `data/qa/...json` for QA examples (`question`, `answers`, etc.),
- `data/corpus/...json` for deduplicated contexts,
- sidecar metadata JSON for provenance/runtime.

### 2) Chunk corpus

#### Raw chunks

```bash
python scripts/build_chunks.py \
  --input_corpus data/corpus/squad_validation_corpus_2000.json \
  --chunking raw \
  --output_path data/chunks/squad_validation_raw_2000.json
```

#### Fixed chunks

```bash
python scripts/build_chunks.py \
  --input_corpus data/corpus/squad_validation_corpus_2000.json \
  --chunking fixed \
  --fixed_chunk_size 120 \
  --fixed_overlap 20 \
  --output_path data/chunks/squad_validation_fixed_2000.json
```

#### Semantic chunks

```bash
python scripts/build_chunks.py \
  --input_corpus data/corpus/squad_validation_corpus_2000.json \
  --chunking semantic \
  --semantic_chunk_size 250 \
  --semantic_threshold 0.75 \
  --output_path data/chunks/squad_validation_semantic_2000.json
```

### 3A) Build local dense index (optional)

```bash
python scripts/build_local_index.py \
  --input_chunks data/chunks/squad_validation_fixed_2000.json \
  --embedding_model BAAI/bge-small-en-v1.5 \
  --output_dir data/index/squad_validation_fixed_2000
```

Output:

- `chunks.json`,
- `embeddings.npy`,
- `metadata.json`.

### 3B) Build Qdrant index (optional)

Create `.env` in project root:

```env
QDRANT_URL=...
QDRANT_API_KEY=...
QDRANT_COLLECTION_NAME=wiki_chunks_bge_small
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
```

Then upload chunks:

```bash
python scripts/build_qdrant_index.py \
  --input_chunks data/chunks/squad_validation_fixed_2000.json \
  --collection_name wiki_chunks_bge_small \
  --embedding_model BAAI/bge-small-en-v1.5 \
  --batch_size 64 \
  --recreate
```

Important: use the same embedding model for both indexing and retrieval.

## Running Experiments (Unified Runner)

Main runner: `scripts/run_experiment.py`

### Closed-book baseline

```bash
python scripts/run_experiment.py \
  --mode closed_book \
  --qa_dataset squad \
  --qa_split validation \
  --num_questions 100 \
  --generator_type simple \
  --generator_model google/flan-t5-base \
  --output_dir results \
  --output_name closed_book_squad_q100.json
```

### BM25 RAG (local chunks)

```bash
python scripts/run_experiment.py \
  --mode bm25 \
  --qa_dataset squad \
  --qa_split validation \
  --num_questions 100 \
  --chunks_path data/chunks/squad_validation_fixed_2000.json \
  --top_k 5 \
  --generator_type simple \
  --generator_model google/flan-t5-base \
  --output_dir results \
  --output_name bm25_squad_fixed_q100_top5.json
```

### Local dense RAG

```bash
python scripts/run_experiment.py \
  --mode local_dense \
  --qa_dataset squad \
  --qa_split validation \
  --num_questions 100 \
  --chunks_path data/index/squad_validation_fixed_2000/chunks.json \
  --embeddings_path data/index/squad_validation_fixed_2000/embeddings.npy \
  --embedding_model BAAI/bge-small-en-v1.5 \
  --top_k 5 \
  --generator_type simple \
  --generator_model google/flan-t5-base \
  --output_dir results \
  --output_name local_dense_squad_fixed_q100_top5.json
```

### Qdrant dense RAG

```bash
python scripts/run_experiment.py \
  --mode qdrant_dense \
  --qa_dataset squad \
  --qa_split validation \
  --num_questions 100 \
  --collection_name wiki_chunks_bge_small \
  --embedding_model BAAI/bge-small-en-v1.5 \
  --top_k 5 \
  --generator_type simple \
  --generator_model google/flan-t5-base \
  --output_dir results \
  --output_name qdrant_dense_squad_q100_top5.json
```

### Using local QA JSON

If `--qa_path` is provided, dataset loading is bypassed:

```bash
python scripts/run_experiment.py \
  --mode bm25 \
  --qa_path data/qa/squad_validation_qa_2000.json \
  --num_questions 100 \
  --chunks_path data/chunks/squad_validation_fixed_2000.json \
  --top_k 5 \
  --output_dir results \
  --output_name bm25_localqa_q100.json
```

## Chunk Size and Attention Diagnostics

Use `scripts/chunk_size_diagnostics.py` for the final-report diagnostics around chunk-size effects and generator attention.

### Retrieval heatmap

This compares whether retrieved chunks contain the gold answer across multiple experiment result files.

```bash
python scripts/chunk_size_diagnostics.py retrieval-heatmap \
  --result_paths results/bm25_fixed20.json results/bm25_fixed70.json results/bm25_fixed120.json \
  --labels fixed20 fixed70 fixed120 \
  --num_questions 100 \
  --output_csv results/retrieval_heatmap_chunk_size.csv \
  --output_png results/retrieval_heatmap_chunk_size.png
```

### Generator cross-attention probe

This extracts FLAN-T5 decoder-to-encoder cross-attention and computes answer attention mass for selected examples.

```bash
python scripts/chunk_size_diagnostics.py cross-attention \
  --result_path results/bm25_fixed70.json \
  --example_indices 7 15 31 59 83 \
  --label_source gold \
  --output_dir results/attention_fixed70 \
  --output_png
```

The resulting CSV files report token-level attention scores and `answer_attention_mass`, the amount of averaged generator cross-attention assigned to gold-answer tokens in the retrieved input.
## Analyze Results

Use `scripts/analyze_experiment.py` subcommands.

### Summary

```bash
python scripts/analyze_experiment.py summary \
  --result_path results/bm25_squad_fixed_q100_top5.json \
  --corpus_path data/corpus/squad_validation_corpus_2000.json \
  --chunks_path data/chunks/squad_validation_fixed_2000.json \
  --qa_dataset squad \
  --qa_split validation \
  --num_questions 100
```

### Inspect per-example outputs

```bash
python scripts/analyze_experiment.py inspect \
  --result_path results/bm25_squad_fixed_q100_top5.json \
  --only_errors \
  --limit 20 \
  --show_retrieved
```

### Retrieval hit/miss analysis

```bash
python scripts/analyze_experiment.py retrieval \
  --result_path results/bm25_squad_fixed_q100_top5.json \
  --show_hits \
  --show_misses \
  --limit 30
```

### Diagnose failure buckets

```bash
python scripts/analyze_experiment.py diagnose \
  --corpus_path data/corpus/squad_validation_corpus_2000.json \
  --result_path results/bm25_squad_fixed_q100_top5.json \
  --qa_dataset squad \
  --qa_split validation \
  --num_questions 100
```

### Compare runs

```bash
python scripts/analyze_experiment.py compare \
  --result_paths \
    results/closed_book_squad_q100.json \
    results/bm25_squad_fixed_q100_top5.json \
    results/local_dense_squad_fixed_q100_top5.json \
  --sort_by avg_f1 \
  --descending
```

Most analysis subcommands also support `--export_csv`.

## Result JSON Schema (Unified Runner)

Each run saved by `run_experiment.py` includes:

- top-level summary metrics (`avg_exact_match`, `avg_f1`, `avg_answer_containment`),
- runtime metadata (`setup_time_sec`, avg per-example timings),
- environment metadata (platform, hostname, device),
- `experiment_config` (all important args),
- `results` list with per-example details:
  - question, gold answers, prediction,
  - EM/F1/containment,
  - retrieval/generation/example latency,
  - retrieved chunks (chunk metadata + score + rank).

## Legacy Scripts

These scripts are still present for quick baselines and historical experiments:

- `scripts/run_closed_book.py`
- `scripts/run_bm25_rag.py`
- `scripts/run_closed_book_batch.py`
- `scripts/run_bm25_rag_batch.py`
- `scripts/run_bm25_corpus_ablation.py`
- `scripts/run_qwen_rag.py`
- `scripts/load_*` helper scripts

Prefer the unified runner for new experiments.

## Known Notes

- `results/` and most of `data/` are git-ignored.
- `data/corpus/wiki_oracle.json` is tracked and used as a small curated corpus.
- semantic chunking uses NLTK sentence tokenization and may download `punkt` at first run.
- Qwen support exists in `src/generation/generator.py` but typically requires a capable GPU.
- Qwen experiments are much slower than FLAN-T5 experiments and may require sufficient GPU memory, Hugging Face model access, and `trust_remote_code=True` support.
- `requirements.txt` currently contains some repeated dependencies from iterative development; it can be cleaned later, but it is still usable for environment setup.

## Suggested Reproducible Baseline

For a clean baseline comparison on SQuAD validation:

1. build `squad` QA+corpus with `sample_size=2000`,
2. build fixed chunks (`120/20`),
3. run `closed_book`, `bm25`, and `local_dense` with same `num_questions`, `top_k`, and generator,
4. compare with `analyze_experiment.py compare`,
5. export CSV summaries for reporting.

## Final Report Reproduction Target

The final report's controlled chunk-size sweep uses:

- Dataset: SQuAD validation subset
- Corpus: deduplicated SQuAD contexts
- Retriever: BM25
- Generator: `google/flan-t5-base`
- Evaluation size: `N=100`
- Retrieval: `top_k=5`
- Chunking: fixed chunks from 10 to 200 tokens with overlap set to roughly one-fifth of the chunk size

The attention diagnostic uses the same q100 setup and computes generator cross-attention on five representative examples (`7, 15, 31, 59, 83`).

## Team

| Name           | Email              |
|----------------|--------------------|
| Mei Aohan      | meiaohan@bu.edu    |
| Haoran Zhang   | zhr114@bu.edu      |
| Haotian Liu    | krisliuu@bu.edu    |
| Jiasong Huang  | hjs1026@bu.edu     |



