from __future__ import annotations

import os
import uuid
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.retrieval.embedder import Embedder

load_dotenv()


class QdrantIndexer:
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

    def collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in collections)

    def create_collection(self, recreate: bool = False) -> None:
        vector_size = self.embedder.get_dimension()

        if self.collection_exists():
            if recreate:
                self.client.delete_collection(collection_name=self.collection_name)
            else:
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def _to_qdrant_id(self, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def _build_points(self, chunks: list[dict[str, Any]]) -> list[PointStruct]:
        texts = [chunk["text"] for chunk in chunks]
        vectors = self.embedder.encode_documents(texts)

        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            points.append(
                PointStruct(
                    id=self._to_qdrant_id(chunk["chunk_id"]),
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "title": chunk["title"],
                        "text": chunk["text"],
                        "chunk_index": chunk["chunk_index"],
                        "method": chunk["method"],
                    },
                )
            )
        return points

    def index_chunks(self, chunks: list[dict[str, Any]], batch_size: int = 64) -> None:
        if not chunks:
            print("No chunks to index.")
            return

        total = len(chunks)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = chunks[start:end]
            points = self._build_points(batch)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            print(f"Indexed {end}/{total} chunks.")

    def count_points(self) -> int:
        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )
        return result.count