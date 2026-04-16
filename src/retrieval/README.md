# Dense Retrieval Module

This module is responsible for:

1. Embedding chunk text into vectors
2. Storing those vectors in Qdrant
3. Retrieving the top-k most relevant chunks for a query
4. Returning the retrieved chunk text to the generator

---

## Input

The retriever expects chunked documents in this format:

```python
{
    "chunk_id": f"{doc_id}_chunk_{chunk_index}",
    "doc_id": doc_id,
    "title": title,
    "text": chunk_text,
    "chunk_index": chunk_index,
    "method": "semantic",
}
```

### Field meanings

- `chunk_id`: unique identifier for the chunk
- `doc_id`: identifier of the original document
- `title`: document title
- `text`: chunk text content
- `chunk_index`: order of the chunk inside the document
- `method`: chunking method, such as `"semantic"` or `"fixed"`

---

## Output

### Output of indexing step

Each chunk is converted into:

- an embedding vector
- a Qdrant point with payload metadata

Conceptually, each stored point looks like this:

```python
{
    "id": "<qdrant_uuid>",
    "vector": [...embedding values...],
    "payload": {
        "chunk_id": "...",
        "doc_id": "...",
        "title": "...",
        "text": "...",
        "chunk_index": 0,
        "method": "semantic",
    }
}
```

### Output of retrieval step

The retriever returns the top-k most relevant chunks in this format:

```python
[
    {
        "chunk_id": "...",
        "doc_id": "...",
        "title": "...",
        "text": "...",
        "chunk_index": 0,
        "method": "semantic",
        "score": 0.8033,
    },
    ...
]
```

The generator should use the returned `text` fields as context.

---

## Qdrant Configuration

This project uses **Qdrant** as the vector database.

### Collection settings

- **Collection name**: configured through `.env`
- **Distance metric**: `COSINE`
- **Vector size**: automatically determined from the embedding model

### Why cosine distance?

The embedding vectors are normalized, so cosine similarity is the correct retrieval metric.

---

## `.env` Configuration

Create a `.env` file in the project root:

```env
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=wiki_chunks_bge_small
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
```

### Environment variables

- `QDRANT_URL`: your Qdrant cluster endpoint
- `QDRANT_API_KEY`: your Qdrant API key
- `QDRANT_COLLECTION_NAME`: collection name used for storing and searching chunk vectors
- `EMBEDDING_MODEL_NAME`: embedding model used for both chunk embeddings and query embeddings

---

## Important Rule

The same embedding model must be used for:

- indexing chunk vectors
- embedding user queries

If the embedding model changes, the collection must be rebuilt.

---

## Dense Retrieval Flow

```text
chunked documents
→ embed chunk text
→ store vectors + payload in Qdrant

user query
→ embed query
→ search Qdrant
→ retrieve top-k chunks
→ send retrieved text to generator
```

---

## Example Usage

### Indexing

```python
from qdrant_indexer import QdrantIndexer

indexer = QdrantIndexer()
indexer.create_collection(recreate=True)
indexer.index_chunks(chunks, batch_size=64)
```

### Retrieval

```python
from dense_retriever import DenseRetriever

retriever = DenseRetriever()
results = retriever.retrieve("Who invented the telephone?", k=5)
```

---

## Notes

- `chunk_id` is preserved in the payload for debugging and tracking
- Qdrant point IDs are stored as UUIDs
- Retrieved vectors are **not** passed to the generator
- Retrieved **text** is passed to the generator