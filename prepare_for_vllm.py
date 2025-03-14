import os
import torch
from pathlib import Path
import argparse
from transformers import LlamaForCausalLM, AutoTokenizer
import json

def convert_to_vllm_compatible(checkpoint_path, output_dir, config_path=None):
    """
    Convert the model to a vLLM-compatible format by removing MAC-specific components
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint['student_state_dict']
    
    # Try to find config file if not provided
    if config_path is None:
        # Look in checkpoint directory first
        checkpoint_dir = Path(checkpoint_path).parent
        config_path = checkpoint_dir / "initial_config.json"
        if not config_path.exists():
            raise ValueError("No config file found. Please specify with --config_path")
    
    # Load the config
    print(f"Using config from {config_path}")
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Create a model with this config - must modify it for standard LlamaForCausalLM
    config_data["architectures"] = ["LlamaForCausalLM"]
    config_data["model_type"] = "llama"
    
    # Remove MAC-specific config
    if "mac_module_config" in config_data:
        del config_data["mac_module_config"]
    
    # Create the model
    model = LlamaForCausalLM.from_config(config_data)
    
    # Extract and load only the Llama parts of the state dict
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('llama.'):
            new_key = key.replace('llama.', '')
            new_state_dict[new_key] = value
    
    # Load the state dict
    model.load_state_dict(new_state_dict, strict=False)
    
    # Save the model
    model.save_pretrained(output_dir)
    
    # Save the vLLM compatible config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    # Copy tokenizer from teacher model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=os.getenv("HF_TOKEN"))
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model converted and saved to {output_dir}")
    print("Note: This version removes the MAC-specific components for vLLM compatibility")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert distilled model to vLLM-compatible format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--output_dir", type=str, default="vllm_model", help="Directory to save the vLLM-compatible model")
    parser.add_argument("--config_path", type=str, default=None, help="Path to the model config file (defaults to initial_config.json in checkpoint dir)")
    
    args = parser.parse_args()
    convert_to_vllm_compatible(args.checkpoint, args.output_dir, args.config_path) 