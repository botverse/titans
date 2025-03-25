import torch
import json
from pathlib import Path
import argparse  # (B, standard library)

def analyze_state_dict(checkpoint_path: str = "checkpoints/checkpoint_epoch_1.pt"):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")  # (B, checkpoint)
    state_dict = checkpoint['student_state_dict']  # (B, dict)
    
    # Categorize keys
    mac_keys = []
    standard_keys = []
    
    for key in state_dict.keys():
        print(key)
    
    # Print analysis
    print(f"Total keys: {len(state_dict)}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect state dict file for a checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch_1.pt",
                        help="Path to checkpoint file to analyze")
    args = parser.parse_args()
    
    analyze_state_dict(args.checkpoint) 