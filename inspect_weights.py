import torch
import numpy as np

# Load and analyze checkpoint weights
checkpoint = torch.load('checkpoints/checkpoint_epoch_1.pt', map_location="cpu")
state_dict = checkpoint['student_state_dict']

# Check embed/lm_head weight statistics
if 'llama.embed_tokens.weight' in state_dict:
    embed = state_dict['llama.embed_tokens.weight']
    print(f"Embeddings: shape={embed.shape}, mean={embed.mean().item():.6f}, std={embed.std().item():.6f}")  # (vocab_size, hidden_size)

if 'lm_head.weight' in state_dict:
    lm_head = state_dict['lm_head.weight']
    print(f"LM head: shape={lm_head.shape}, mean={lm_head.mean().item():.6f}, std={lm_head.std().item():.6f}")  # (vocab_size, hidden_size)
    
# Check for NaN values
has_nan = False
for name, tensor in state_dict.items():
    if torch.isnan(tensor).any():
        has_nan = True
        print(f"NaN values found in {name}")

print(f"Model has NaN values: {has_nan}")
