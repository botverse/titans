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
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )
    
    # Extract state dict
    state_dict = checkpoint['student_state_dict']
    
    # Transform the state dict to match HuggingFace Llama format
    new_state_dict = {}
    
    # Track which norm weights were properly transferred
    norm_layers_transferred = set()
    
    for key, value in state_dict.items():
        # Skip MAC-specific parts
        if key.startswith('mac_module.'):
            continue
            
        # Ensure we keep full precision
        value = value.float()  # Convert to full precision
        
        if key.startswith('llama.'):
            # Extract the relevant part after 'llama.'
            suffix = key[len('llama.'):]
            
            # Handle layernorm weights specially to ensure they're preserved
            if 'layernorm' in key or 'norm' in key:
                if key.startswith('llama.model.'):
                    new_key = key.replace('llama.model.', 'model.')
                else:
                    new_key = f"model.{suffix}"
                norm_layers_transferred.add(new_key)
            # Regular handling for other keys
            elif suffix.startswith('model.'):
                new_key = f"model.{suffix[len('model.'):]}"
            else:
                new_key = f"model.{suffix}"
                
            new_state_dict[new_key] = value
            
        elif key.startswith('lm_head.'):
            # Keep lm_head keys as is
            new_state_dict[key] = value
    
    # Save the config to the output directory
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    # Save the state dict directly to avoid any automatic processing
    print(f"Saving state dict with {len(new_state_dict)} keys to {output_dir / 'pytorch_model.bin'}")
    torch.save(new_state_dict, output_dir / "pytorch_model.bin")
    
    # Copy tokenizer from teacher model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=os.getenv("HF_TOKEN"))
    tokenizer.save_pretrained(output_dir)
    
    # Create a generation config
    generation_config = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "max_length": 2048
    }
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(generation_config, f, indent=2)
    
    # Check for norm layers
    print(f"Transferred {len(norm_layers_transferred)} normalization layers")
    
    print(f"Model converted and saved to {output_dir}")
    print("Note: This version removes the MAC-specific components for vLLM compatibility")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert distilled model to vLLM-compatible format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--output_dir", type=str, default="vllm_model", help="Directory to save the vLLM-compatible model")
    parser.add_argument("--config_path", type=str, default=None, help="Path to the model config file (defaults to initial_config.json in checkpoint dir)")
    
    args = parser.parse_args()
    convert_to_vllm_compatible(args.checkpoint, args.output_dir, args.config_path) 