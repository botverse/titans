import os
import torch
from pathlib import Path
import argparse
from transformers import AutoTokenizer
import json
import shutil

def export_model_to_hf_format(checkpoint_path, output_dir, config_path=None):
    """
    Export the trained model to a HuggingFace compatible format
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Extract model state dict
    state_dict = checkpoint['student_state_dict']
    
    # Save the complete state dict, including MAC parameters
    torch.save(state_dict, output_dir / "pytorch_model.bin")
    
    # Try to find config file if not provided
    if config_path is None:
        # Look in checkpoint directory first
        checkpoint_dir = Path(checkpoint_path).parent
        config_path = checkpoint_dir / "initial_config.json"
        if not config_path.exists():
            raise ValueError("No config file found. Please specify with --config_path")
    
    # Load and save the config
    print(f"Using config from {config_path}")
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Make sure architectures and model_type are set correctly
    config_data["architectures"] = ["MACTransformer"]
    config_data["model_type"] = "llama_mac"
    
    # Save the config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    # Copy the model definition file
    shutil.copy("models/llama_titans.py", output_dir / "modeling_llama_mac.py")
    
    # Save tokenizer 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=os.getenv("HF_TOKEN"))
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model with MAC module exported to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export distilled model to HuggingFace format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--output_dir", type=str, default="exported_model", help="Directory to save the exported model")
    parser.add_argument("--config_path", type=str, default=None, help="Path to the model config file (defaults to initial_config.json in checkpoint dir)")
    
    args = parser.parse_args()
    export_model_to_hf_format(args.checkpoint, args.output_dir, args.config_path) 