import torch
import json
from pathlib import Path

def analyze_state_dict(checkpoint_path: str = "checkpoints/checkpoint_epoch_1.pt"):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint['student_state_dict']
    
    # Categorize keys
    mac_keys = []
    standard_keys = []
    
    for key in state_dict.keys():
        if "mac_module" in key:
            mac_keys.append(key)
        else:
            standard_keys.append(key)
    
    # Print analysis
    print(f"Total keys: {len(state_dict)}")
    print(f"MAC-specific keys: {len(mac_keys)}")
    print(f"Standard LLaMA keys: {len(standard_keys)}\n")
    
    # Print MAC module structure
    print("MAC Module Structure:")
    for key in mac_keys:
        parts = key.split(".")
        indent = "  " * (parts.index("mac_module") + 1)
        print(f"{indent}{' → '.join(parts[parts.index('mac_module'):])}")

    # Print standard key structure
    print("\nStandard LLaMA Structure:")
    for key in standard_keys[:10]:  # First 10 for brevity
        parts = key.split(".")
        indent = "  " * (len(parts) - 1)
        print(f"{indent}{parts[-1]}")
    
    # Save full lists
    analysis = {
        "mac_keys": mac_keys,
        "standard_keys": standard_keys,
        "key_count": len(state_dict),
        "mac_key_count": len(mac_keys),
        "standard_key_count": len(standard_keys)
    }
    
    with open("state_dict_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

if __name__ == "__main__":
    analyze_state_dict() 