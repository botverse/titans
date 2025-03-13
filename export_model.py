import os
import torch
from pathlib import Path
import argparse
from transformers import LlamaConfig, AutoTokenizer
from models.llama_titans import MACTransformer, MACModule
import json
import shutil

def infer_or_extract_config(state_dict):
    """
    Infer model configuration from state dictionary
    """
    print("Inferring config from model structure...")
    
    # Get hidden size from embedding weights
    if 'llama.model.embed_tokens.weight' in state_dict:
        vocab_size, hidden_size = state_dict['llama.model.embed_tokens.weight'].shape  # (vocab_size, C)
    else:
        embed_key = next((k for k in state_dict.keys() if 'embed_tokens.weight' in k), None)
        if embed_key:
            vocab_size, hidden_size = state_dict[embed_key].shape  # (vocab_size, C)
        else:
            vocab_size = 32000
            hidden_size = 2048
            print("Warning: Could not infer vocab_size and hidden_size from model")
    
    # Count number of layers
    layer_pattern = 'llama.model.layers.'
    layer_keys = [k for k in state_dict.keys() if layer_pattern in k]
    layer_numbers = set()
    for key in layer_keys:
        parts = key.split(layer_pattern)[1].split('.')
        if parts and parts[0].isdigit():
            layer_numbers.add(int(parts[0]))
    
    num_hidden_layers = len(layer_numbers) if layer_numbers else 16
    
    # Get number of attention heads from q_proj weight
    q_proj_key = next((k for k in state_dict.keys() if 'q_proj.weight' in k), None)
    if q_proj_key and hidden_size > 0:
        out_dim = state_dict[q_proj_key].shape[0]  # (num_heads*head_dim, C)
        
        # Improved head_dim calculation logic
        # In LLaMA, typically head_dim = hidden_size / num_heads
        # So we can estimate num_heads by looking at q_proj dimensions
        if out_dim % hidden_size == 0:
            # For special case where out_dim is a multiple of hidden_size
            num_attention_heads = out_dim // hidden_size
            head_dim = hidden_size
        else:
            # Try common head dimensions (128, 64, etc.)
            for possible_head_dim in [128, 64, 32, 16]:
                if out_dim % possible_head_dim == 0:
                    num_attention_heads = out_dim // possible_head_dim
                    head_dim = possible_head_dim
                    break
            else:
                # Fallback: Simple approximation
                num_attention_heads = max(1, out_dim // hidden_size)
                head_dim = hidden_size // num_attention_heads
        
        # Check for Multi-Query Attention (MQA) or Grouped-Query Attention (GQA)
        kv_proj_key = next((k for k in state_dict.keys() if 'k_proj.weight' in k), None)
        if kv_proj_key:
            kv_out_dim = state_dict[kv_proj_key].shape[0]  # (num_kv_heads*head_dim, C)
            num_key_value_heads = max(1, kv_out_dim // head_dim)  # Ensure at least 1
        else:
            num_key_value_heads = max(1, num_attention_heads)  # Ensure at least 1
    else:
        # Fallback values
        num_attention_heads = 16
        num_key_value_heads = 4  # Standard GQA ratio for LLaMA-3
        head_dim = hidden_size // num_attention_heads
        print("Warning: Could not infer num_attention_heads from model")
    
    # Debug output to help understand the inference
    print(f"Head inference: out_dim={out_dim if q_proj_key else 'unknown'}, " 
          f"head_dim={head_dim}, num_heads={num_attention_heads}, num_kv_heads={num_key_value_heads}")
    
    # Get intermediate size from MLP
    gate_proj_key = next((k for k in state_dict.keys() if 'gate_proj.weight' in k), None)
    if gate_proj_key:
        intermediate_size = state_dict[gate_proj_key].shape[0]  # (intermediate_size, C)
    else:
        intermediate_size = hidden_size * 2
        print("Warning: Could not infer intermediate_size from model")
    
    # Create config
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=2048,  # Default fallback
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None
    )
    
    print(f"Inferred config: hidden_size={hidden_size}, layers={num_hidden_layers}, heads={num_attention_heads}")
    return config

def export_model_to_hf_format(checkpoint_path, output_dir):
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
    
    # Save config with MAC module information
    config = infer_or_extract_config(state_dict)
    
    # Add MAC-specific configuration
    config.mac_module_config = {
        "num_persistent": int(state_dict["mac_module.persistent_memory"].shape[0]),
        "memory_size": int(state_dict["mac_module.long_term_memory"].shape[0]),
        "alpha": 0.1  # Get from state_dict if available
    }
    
    # Save the custom config
    config.save_pretrained(output_dir)
    
    # Copy the model definition file
    shutil.copy("models/llama_titans.py", output_dir / "modeling_llama_mac.py")
    
    # Create a model config file that specifies how to load the custom model
    with open(output_dir / "config.json", "r") as f:
        config_data = json.load(f)
    
    config_data["architectures"] = ["MACTransformer"]
    config_data["model_type"] = "llama_mac"
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=os.getenv("HF_TOKEN"))
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model with MAC module exported to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export distilled model to HuggingFace format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--output_dir", type=str, default="exported_model", help="Directory to save the exported model")
    
    args = parser.parse_args()
    export_model_to_hf_format(args.checkpoint, args.output_dir) 