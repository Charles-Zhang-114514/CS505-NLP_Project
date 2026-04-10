# CS505 NLP Project — Modular RAG System

## Overview

This project implements a modular Retrieval-Augmented Generation (RAG) system and evaluates how different retrieval and generation strategies affect question answering performance.

The goal is to compare modern RAG components with simpler baselines, and understand how retrieval improves factual accuracy.

---

## Current Progress (Baseline)

We have implemented two baseline systems:

### 1. Closed-book QA
- The model answers questions without any external knowledge
- Uses `flan-t5-base` as the generator
- Relies entirely on parametric knowledge

### 2. BM25 + Generation (RAG)
- Uses BM25 for document retrieval (sparse retrieval)
- Retrieved documents are provided as context to the generator
- Demonstrates how retrieval improves answer quality

---

## Project Structure


project/
│
├── data/ # datasets (to be added later)
├── results/ # experiment outputs
│
├── scripts/ # runnable scripts
│ ├── run_closed_book.py # closed-book baseline
│ └── run_bm25_rag.py # BM25 + RAG baseline
│
├── src/
│ ├── chunking/ # chunking methods (future work)
│ │ └── fixed_chunk.py
│ │
│ ├── retrieval/ # retrieval modules
│ │ └── bm25_retriever.py
│ │
│ ├── generation/ # generation model
│ │ └── generator.py
│ │
│ └── eval/ # evaluation metrics
│ └── qa_metrics.py
│
├── requirements.txt
└── README.md


---

## How to Run

### 1. Install dependencies
pip install -r requirements.txt
2. Run closed-book baseline
python scripts/run_closed_book.py
3. Run BM25 RAG baseline
python scripts/run_bm25_rag.py
Example Result

For the question:

Who wrote Pride and Prejudice?

Closed-book output:

Incorrect answer

BM25 + RAG output:

Correct answer: Jane Austen

This shows that retrieval improves factual accuracy.

Components
Generator (generator.py)
Loads pretrained model (flan-t5-base)
Supports:
Closed-book QA
Context-based QA (RAG)
Retriever (bm25_retriever.py)
Implements BM25 retrieval
Returns top-k relevant documents
Evaluation (qa_metrics.py)
Exact Match (EM)
F1 Score
Scripts
run_closed_book.py: runs closed-book baseline
run_bm25_rag.py: runs BM25 + RAG pipeline
Notes
Current implementation uses a toy corpus for testing
The system is modular and can be extended easily
Future Work
Dense retrieval (embedding-based)
Real dataset (NQ Open)
Chunking strategies (fixed vs semantic)
Larger-scale evaluation (SCC cluster)
Authors

Boston University — CS505 NLP Project