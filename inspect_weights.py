import torch
from pathlib import Path

def find_latest_run():
    runs_dir = Path("runs")
    experiments = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("distil_")]
    return max(experiments, key=lambda x: x.stat().st_mtime) if experiments else None

latest_run = find_latest_run()
model_dir = latest_run / "vllm_llama_model" if latest_run else None
model_path = model_dir / "pytorch_model.bin"

# Load and analyze weights
state_dict = torch.load(model_path, map_location="cpu")

# Check key structure
print("Model keys:")
for key in state_dict.keys():
    print(f"- {key}")

# Check embed/lm_head weight statistics
if 'model.embed_tokens.weight' in state_dict:
    embed = state_dict['model.embed_tokens.weight']
    print(f"\nEmbeddings: shape={embed.shape}, mean={embed.mean().item():.6f}, std={embed.std().item():.6f}")

if 'lm_head.weight' in state_dict:
    lm_head = state_dict['lm_head.weight']
    print(f"LM head: shape={lm_head.shape}, mean={lm_head.mean().item():.6f}, std={lm_head.std().item():.6f}")
    
# Check for NaN values
has_nan = False
for name, tensor in state_dict.items():
    if torch.isnan(tensor).any():
        has_nan = True
        print(f"NaN values found in {name}")

print(f"\nModel has NaN values: {has_nan}")
