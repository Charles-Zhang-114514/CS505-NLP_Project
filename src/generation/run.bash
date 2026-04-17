python -c "
from src.generation.generator import QwenGenerator
gen = QwenGenerator()
result = gen.generate('Who wrote Pride and Prejudice?', [{'chunk_id':'1','doc_id':'d1','text':'Pride and Prejudice is a novel by Jane Austen, published in 1813.','score':0.9,'rank':1}])
print('Answer:', result['answer'])
"