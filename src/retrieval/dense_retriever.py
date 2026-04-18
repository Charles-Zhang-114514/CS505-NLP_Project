from __future__ import annotations

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from src.retrieval.embedder import Embedder

load_dotenv()


class DenseRetriever:
    def __init__(
        self,
        collection_name: str | None = None,
        model_name: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ):
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME")
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME")
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")

        if not self.collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME is not set.")
        if not self.model_name:
            raise ValueError("EMBEDDING_MODEL_NAME is not set.")
        if not self.url:
            raise ValueError("QDRANT_URL is not set.")
        if not self.api_key:
            raise ValueError("QDRANT_API_KEY is not set.")

        self.embedder = Embedder(model_name=self.model_name)
        self.client = QdrantClient(url=self.url, api_key=self.api_key)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        query_vec = self.embedder.encode_query(query).tolist()

        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vec,
                limit=k,
                with_payload=True,
            )
        except AttributeError:
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=k,
                with_payload=True,
            ).points

        results = []
        for hit in hits:
            payload = hit.payload
            results.append(
                {
                    "chunk_id": payload["chunk_id"],
                    "doc_id": payload["doc_id"],
                    "title": payload["title"],
                    "text": payload["text"],
                    "chunk_index": payload["chunk_index"],
                    "method": payload["method"],
                    "score": float(hit.score),
                }
            )

        return results