import torch

def inspect_model_weights(path):
    """Check if model weights are in a reasonable distribution"""
    state_dict = torch.load(path, map_location="cpu")
    
    if isinstance(state_dict, dict) and "student_state_dict" in state_dict:
        state_dict = state_dict["student_state_dict"]
    
    # Sample some key layers
    sample_layers = []
    for key in state_dict:
        if "embed_tokens" in key or "lm_head" in key or "layer.0" in key:
            sample_layers.append(key)
    
    print(f"Examining {len(sample_layers)} sample layers...")
    for key in sample_layers:
        tensor = state_dict[key]
        stats = {
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "shape": tensor.shape
        }
        print(f"{key}: {stats}")
    
    # Check for abnormal values
    has_issues = False
    for key, tensor in state_dict.items():
        if torch.isnan(tensor).any():
            print(f"WARNING: NaN values in {key}")
            has_issues = True
        elif torch.abs(tensor).max() > 100:
            print(f"WARNING: Unusually large values in {key}: {torch.abs(tensor).max().item()}")
            has_issues = True
    
    return not has_issues

# Check original checkpoint and exported model
original_ok = inspect_model_weights("checkpoints/checkpoint_epoch_1.pt")
exported_ok = inspect_model_weights("vllm_model/pytorch_model.bin")

print(f"Original checkpoint status: {'OK' if original_ok else 'Has issues'}")
print(f"Exported model status: {'OK' if exported_ok else 'Has issues'}")

