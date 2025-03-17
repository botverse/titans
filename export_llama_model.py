import os
import torch
from pathlib import Path
import argparse
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig
import json

def convert_to_vllm_compatible(checkpoint_path, output_dir, config_path=None):
    """
    Convert the model to a vLLM-compatible format by removing MAC-specific components
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config properly
    if config_path is None:
        config_path = Path(checkpoint_path).parent / "initial_config.json"
    
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    # Reset architecture to standard Llama
    config_data["architectures"] = ["LlamaForCausalLM"]
    config_data["model_type"] = "llama"  # Ensure standard model type
    
    # Create config object and model
    config = LlamaConfig.from_dict(config_data)
    model = LlamaForCausalLM._from_config(config)  # Use protected method
    
    # Load state dict
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )
    
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