import os
import torch
from pathlib import Path
import argparse
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
import json

def convert_to_vllm_compatible(checkpoint_path, output_dir):
    """
    Convert the model to a vLLM-compatible format by removing MAC-specific components
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint['student_state_dict']
    
    # Infer config from model structure
    print("Inferring config from model structure...")
    # Get hidden size from first weight matrix dimension
    if 'llama.model.embed_tokens.weight' in state_dict:
        vocab_size, hidden_size = state_dict['llama.model.embed_tokens.weight'].shape
    else:
        # Find embed_tokens in the state dict keys
        embed_key = next((k for k in state_dict.keys() if 'embed_tokens.weight' in k), None)
        if embed_key:
            vocab_size, hidden_size = state_dict[embed_key].shape
        else:
            # Fallback to a default
            vocab_size = 32000
            hidden_size = 2048
            print("Warning: Could not infer vocab_size and hidden_size from model")
    
    # Count number of layers by looking for layer pattern
    layer_pattern = 'llama.model.layers.'
    layer_keys = [k for k in state_dict.keys() if layer_pattern in k]
    layer_numbers = set()
    for key in layer_keys:
        # Extract the layer number from the key
        parts = key.split(layer_pattern)[1].split('.')
        if parts and parts[0].isdigit():
            layer_numbers.add(int(parts[0]))
    
    num_hidden_layers = len(layer_numbers) if layer_numbers else 16
    
    # Get number of attention heads from q_proj weight dimensions
    q_proj_key = next((k for k in state_dict.keys() if 'q_proj.weight' in k), None)
    if q_proj_key and hidden_size > 0:
        out_dim = state_dict[q_proj_key].shape[0]
        head_dim = hidden_size // (out_dim // hidden_size)
        num_attention_heads = out_dim // head_dim
        num_key_value_heads = num_attention_heads  # Assuming same number for now
    else:
        num_attention_heads = 16
        num_key_value_heads = 16
        print("Warning: Could not infer num_attention_heads from model")
    
    # Get intermediate size from gate_proj dimensions
    gate_proj_key = next((k for k in state_dict.keys() if 'gate_proj.weight' in k), None)
    if gate_proj_key:
        intermediate_size = state_dict[gate_proj_key].shape[0]
    else:
        intermediate_size = hidden_size * 2  # Common ratio in Llama models
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
        rms_norm_eps=1e-5
    )
    
    print(f"Inferred config: {config}")
    
    # Create a new Llama model
    model = LlamaForCausalLM(config)
    
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
    
    # Make sure the config file has the correct model type and architecture
    with open(output_dir / "config.json", "r") as f:
        config_data = json.load(f)
    
    # Override architecture and model type for standard LLaMA
    config_data["architectures"] = ["LlamaForCausalLM"]
    config_data["model_type"] = "llama"
    
    # Remove any MAC-specific config elements
    if "mac_module_config" in config_data:
        del config_data["mac_module_config"]
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    # Copy tokenizer from teacher model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=os.getenv("HF_TOKEN"))
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model converted and saved to {output_dir}")
    print("Note: This version removes the MAC-specific components for vLLM compatibility")
    print("For full MAC functionality, a custom vLLM implementation would be required")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert distilled model to vLLM-compatible format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--output_dir", type=str, default="vllm_model", help="Directory to save the vLLM-compatible model")
    
    args = parser.parse_args()
    convert_to_vllm_compatible(args.checkpoint, args.output_dir) 