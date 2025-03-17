import torch
import os
from safetensors.torch import load_file

def inspect_model_weights(path):
    """Check if model weights are in a reasonable distribution"""
    # Check if file exists
    if not os.path.exists(path):
        # Try alternative path format
        if path.endswith("pytorch_model.bin"):
            alternative_path = path.replace("pytorch_model.bin", "model.safetensors")
            if os.path.exists(alternative_path):
                print(f"PyTorch model not found, using SafeTensors model at: {alternative_path}")
                path = alternative_path
            else:
                raise FileNotFoundError(f"Neither {path} nor {alternative_path} exists")
    
    # Load state dict based on file type
    if path.endswith(".safetensors"):
        state_dict = load_file(path)
    else:
        state_dict = torch.load(path, map_location="cpu")
    
    if isinstance(state_dict, dict) and "student_state_dict" in state_dict:
        state_dict = state_dict["student_state_dict"]
    
    # Sample some key layers
    sample_layers = []
    for key in list(state_dict.keys())[:10]:  # Just check first 10 keys
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
    
    print(f"Total number of parameters: {sum(p.numel() for p in state_dict.values())}")
    print(f"Key prefix examples: {list(state_dict.keys())[:3]}")
    
    return not has_issues

# Check original checkpoint
print("Checking original checkpoint:")
original_ok = inspect_model_weights("checkpoints/checkpoint_epoch_1.pt")

# Check exported model - will find model.safetensors automatically
print("\nChecking exported model:")
exported_ok = inspect_model_weights("vllm_model/pytorch_model.bin")  # Will try model.safetensors if needed

print(f"\nOriginal checkpoint status: {'OK' if original_ok else 'Has issues'}")
print(f"Exported model status: {'OK' if exported_ok else 'Has issues'}")

