from src.generation.generator import QwenGenerator

if __name__ == "__main__":
    gen = QwenGenerator()

    # 测试 RAG 问答
    result = gen.generate(
        query="Who wrote Pride and Prejudice?",
        retrieved_chunks=[
            {
                "chunk_id": "1",
                "doc_id": "d1",
                "text": "Pride and Prejudice is a novel by Jane Austen, published in 1813.",
                "score": 0.9,
                "rank": 1,
            }
        ],
    )
    print("=== RAG 问答 ===")
    print("Answer:", result["answer"])
    print("Context used:", result["context_used"])

    # 测试 Closed-book
    result2 = gen.answer_closed_book("Who wrote Pride and Prejudice?")
    print("\n=== Closed-book ===")
    print("Answer:", result2["answer"])