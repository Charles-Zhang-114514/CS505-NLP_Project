from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class SimpleGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        # Load tokenizer and model
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def answer_question(self, question: str, max_new_tokens: int = 32) -> str:
        # Build a simple closed-book prompt
        prompt = f"Answer the question briefly.\nQuestion: {question}\nAnswer:"
        return self._generate(prompt, max_new_tokens)

    def answer_with_context(
        self,
        question: str,
        context_docs: list[str],
        max_new_tokens: int = 32
    ) -> str:
        # Join retrieved documents
        context = "\n".join(context_docs)

        # Build a simple RAG prompt
        prompt = (
            f"Answer the question using the context.\n"
            f"Context:\n{context}\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return self._generate(prompt, max_new_tokens)

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate output
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

        # Decode output
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()