from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()