from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, documents: list[str]):
        # Save raw documents
        self.documents = documents

        # Tokenize documents
        self.tokenized_docs = [doc.lower().split() for doc in documents]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query: str, top_k: int = 2) -> list[str]:
        # Tokenize query
        tokenized_query = query.lower().split()

        # Get scores
        scores = self.bm25.get_scores(tokenized_query)

        # Sort documents by score
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        # Return top-k documents
        top_docs = [self.documents[i] for i in ranked_indices[:top_k]]
        return top_docs