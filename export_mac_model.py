import os
import torch
from pathlib import Path
import argparse
from transformers import AutoTokenizer
import json
import shutil
import dotenv

dotenv.load_dotenv()

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
    
    # Separate the state dict by component for easier handling
    llama_state_dict = {}
    mac_state_dict = {}
    lm_head_state_dict = {}
    
    for key, value in state_dict.items():
        # Direct key mapping without prefixes
        if key.startswith('llama.'):
            new_key = key.replace('llama.', '', 1)
            llama_state_dict[new_key] = value
        elif key.startswith('mac_module.'):
            mac_state_dict[key] = value
        elif key.startswith('lm_head.'):
            lm_head_state_dict[key] = value
    
    # Combine all components
    combined_state_dict = {}
    combined_state_dict.update(llama_state_dict)
    combined_state_dict.update(mac_state_dict)
    combined_state_dict.update(lm_head_state_dict)
    
    # Validate key structure
    print("Validating exported keys:")
    expected_prefixes = {'model.', 'mac_module.', 'lm_head.'}
    for key in combined_state_dict:
        if not any(key.startswith(p) for p in expected_prefixes):
            print(f"Unexpected key prefix: {key}")
    
    # Save the transformed state dict
    torch.save(combined_state_dict, output_dir / "pytorch_model.bin")
    
    # Try to find config file if not provided
    if config_path is None:
        # Look in checkpoint directory first
        checkpoint_dir = Path(checkpoint_path).parent
        config_path = checkpoint_dir / "initial_config.json"
        if not config_path.exists():
            raise ValueError("No config file found. Please specify with --config_path")
    
    # Load and update the config
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Ensure MAC module config exists
    if "mac_module_config" not in config_data:
        config_data["mac_module_config"] = {
            "num_persistent": 16,
            "memory_size": 1024, 
            "alpha": 0.1
        }
    
    # Make sure architectures and model_type are set correctly
    config_data["architectures"] = ["MACLlamaForCausalLM"]
    config_data["model_type"] = "llama_mac"
    
    # Save the config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    # Copy the model definition files
    os.makedirs(output_dir / "models", exist_ok=True)
    shutil.copy("models/llama_titans.py", output_dir / "models/llama_titans.py")
    shutil.copy("models/vllm_mac_model.py", output_dir / "models/vllm_mac_model.py") 
    
    # Create a small init file
    with open(output_dir / "models/__init__.py", "w") as f:
        f.write("# This file is needed for module imports\n")
    
    # Save tokenizer 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=os.getenv("HF_TOKEN"))
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model with MAC module exported to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export distilled model to HuggingFace format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--output_dir", type=str, default="vllm_mac_model", help="Directory to save the exported model")
    parser.add_argument("--config_path", type=str, default=None, help="Path to the model config file (defaults to initial_config.json in checkpoint dir)")
    
    args = parser.parse_args()
    export_model_to_hf_format(args.checkpoint, args.output_dir, args.config_path) 