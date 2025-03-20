import os
import torch
from pathlib import Path
import argparse
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig
import json
import dotenv

dotenv.load_dotenv()

def find_latest_run():
    """Find the most recent experiment directory"""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    
    # Find all experiment directories
    experiments = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("distil_")]
    if not experiments:
        return None
    
    # Sort by creation time and return the latest
    return max(experiments, key=lambda x: x.stat().st_mtime)

def find_latest_checkpoint(run_dir):
    """Find the latest checkpoint in a run directory"""
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    
    # Find all checkpoint files
    checkpoints = [f for f in checkpoint_dir.glob("checkpoint_epoch_*.pt")]
    if not checkpoints:
        return None
    
    # Sort by epoch number and return the latest
    return max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))

def convert_to_vllm_compatible(run_dir=None, checkpoint_file=None, config_path=None):
    """
    Convert the model to a vLLM-compatible format by removing MAC-specific components
    """
    # Find latest run if not specified
    if run_dir is None:
        run_dir = find_latest_run()
        if run_dir is None:
            raise ValueError("No experiment runs found in runs directory")
    else:
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise ValueError(f"Run directory {run_dir} does not exist")
    
    # Find latest checkpoint if not specified
    if checkpoint_file is None:
        checkpoint_file = find_latest_checkpoint(run_dir)
        if checkpoint_file is None:
            raise ValueError(f"No checkpoints found in {run_dir}/checkpoints")
    else:
        checkpoint_file = run_dir / "checkpoints" / checkpoint_file
        if not checkpoint_file.exists():
            raise ValueError(f"Checkpoint file {checkpoint_file} does not exist")

    # Set output directory within the run directory
    output_dir = run_dir / "vllm_llama_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use initial_config.json from checkpoints directory if config_path not specified
    if config_path is None:
        config_path = run_dir / "checkpoints" / "initial_config.json"
    
    print(f"Converting model from {checkpoint_file} to vLLM format")
    print(f"Output directory: {output_dir}")
    
    # Load config properly
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    # Reset architecture to standard Llama
    config_data["architectures"] = ["LlamaForCausalLM"]
    config_data["model_type"] = "llama"  # Ensure standard model type
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=True
    )
    
    # Extract state dict
    state_dict = checkpoint['student_state_dict']
    
    # Transform the state dict to match HuggingFace Llama format
    new_state_dict = {}
    for key, value in state_dict.items():
        # Skip MAC-specific parts
        if key.startswith('mac_module.'):
            continue
            
        # Ensure we keep full precision
        value = value.float()  # Convert to full precision
        
        # Correctly handle the double 'model.model.' prefix
        if key.startswith('model.model.'):
            new_key = 'model.' + key[len('model.model.'):]  # remove extra 'model.' prefix
        elif key.startswith('model.'):
            new_key = key  # already correct
        elif key.startswith('llama.'):
            new_key = f"{key[len('llama.'):]}"
        else:
            new_key = f"model.{key}"

        # Handle layer norm weights properly
        if any(x in new_key.lower() for x in ['layernorm', 'norm']):
            # Initialize layer norm weights to 1.0 and ensure proper shape
            if value.dim() == 1:  # It should be a 1D tensor
                value = torch.ones_like(value)
            else:
                print(f"Warning: Unexpected shape for layer norm weight: {key} - {value.shape}")
        
        # Ensure no NaN values
        if torch.isnan(value).any():
            print(f"Warning: NaN values found in {key}, replacing with ones")
            value = torch.ones_like(value)
            
        new_state_dict[new_key] = value

    # Ensure lm_head is properly handled
    if 'model.embed_tokens.weight' in new_state_dict and 'lm_head.weight' not in new_state_dict:
        print("Creating lm_head.weight from embedding weights")
        new_state_dict['lm_head.weight'] = new_state_dict['model.embed_tokens.weight'].clone()

    # Verify key structure
    print("\nVerifying state dict keys:")
    for key in sorted(new_state_dict.keys()):
        print(f"- {key}")

    # Save the transformed state dict
    print(f"\nSaving state dict with {len(new_state_dict)} keys")
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
    
    print(f"Model converted and saved to {output_dir}")
    print("Note: This version removes the MAC-specific components for vLLM compatibility")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert distilled model to vLLM-compatible format")
    parser.add_argument("--run", type=str, help="Run directory name (defaults to latest)")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file name (defaults to latest)")
    parser.add_argument("--config_path", type=str, help="Path to the model config file (defaults to initial_config.json in checkpoint dir)")
    
    args = parser.parse_args()
    
    # Convert paths if provided
    run_dir = Path("runs") / args.run if args.run else None
    
    convert_to_vllm_compatible(run_dir, args.checkpoint, args.config_path) 