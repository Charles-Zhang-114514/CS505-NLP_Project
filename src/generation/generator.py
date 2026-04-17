import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenGenerator:
    def __init__(self, model_name="Qwen/Qwen3.5-4B", enable_thinking=False):
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        self.model.eval()

    def generate(self, query, retrieved_chunks, max_new_tokens=64):
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.get("rank", 0))
        context_texts = [chunk["text"] for chunk in sorted_chunks]
        prompt = self._build_rag_prompt(query, context_texts)
        answer = self._call_model(prompt, max_new_tokens)
        return {"answer": answer, "query": query, "context_used": context_texts, "model": self.model_name}

    def answer_closed_book(self, query, max_new_tokens=64):
        prompt = f"Answer the following question briefly.\nQuestion: {query}\nAnswer:"
        answer = self._call_model(prompt, max_new_tokens)
        return {"answer": answer, "query": query, "context_used": [], "model": self.model_name}

    def _build_rag_prompt(self, query, context_texts):
        context = "\n\n".join(f"[{i+1}] {text}" for i, text in enumerate(context_texts))
        return f"Use the following context to answer the question accurately and briefly.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    def _call_model(self, prompt, max_new_tokens):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
