from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.retrieval.embedder import Embedder


class LocalDenseRetriever:
    def __init__(
        self,
        chunks_path: str,
        embeddings_path: str,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks: list[dict[str, Any]] = json.load(f)
        self.embeddings = np.load(embeddings_path)

        if len(self.chunks) != len(self.embeddings):
            raise ValueError("chunks and embeddings size mismatch")

        self.embedder = Embedder(model_name=model_name)

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        query_vec = self.embedder.encode_query(query)
        # embeddings are normalized in Embedder; cosine = dot product
        scores = self.embeddings @ query_vec
        top_idx = np.argsort(scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            item = dict(self.chunks[int(idx)])
            item["score"] = float(scores[int(idx)])
            item["rank"] = rank
            results.append(item)
        return results
