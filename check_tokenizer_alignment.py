# Compare tokenization between original and exported model
import torch
from transformers import AutoTokenizer
import dotenv
import os
from pathlib import Path

dotenv.load_dotenv()

def find_latest_run():
    runs_dir = Path("runs")
    experiments = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("distil_")]
    return max(experiments, key=lambda x: x.stat().st_mtime) if experiments else None

latest_run = find_latest_run()
model_dir = latest_run / "vllm_llama_model" if latest_run else None

original_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=os.getenv("HF_TOKEN"))
exported_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

prompt = "what's the capital of france?"
original_tokens = original_tokenizer.encode(prompt)
exported_tokens = exported_tokenizer.encode(prompt)

print(f"Original tokens: {original_tokens}")
print(f"Original decoded: {original_tokenizer.decode(original_tokens)}")
print(f"Exported tokens: {exported_tokens}")
print(f"Exported decoded: {exported_tokenizer.decode(exported_tokens)}")
print(f"Match: {original_tokens == exported_tokens}")
