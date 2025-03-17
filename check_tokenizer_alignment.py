# Compare tokenization between original and exported model
import torch
from transformers import AutoTokenizer
import dotenv
dotenv.load_dotenv()

original_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
exported_tokenizer = AutoTokenizer.from_pretrained("vllm_model/")

prompt = "what's the capital of france?"
original_tokens = original_tokenizer.encode(prompt)
exported_tokens = exported_tokenizer.encode(prompt)

print(f"Original tokens: {original_tokens}")
print(f"Exported tokens: {exported_tokens}")
print(f"Match: {original_tokens == exported_tokens}")
