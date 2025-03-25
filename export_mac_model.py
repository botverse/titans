import os
import torch
from pathlib import Path
import argparse
from transformers import AutoTokenizer
import json
import shutil
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

def export_model_to_hf_format(run_dir=None, checkpoint_file=None, config_path=None):
    """
    Export the trained model to a HuggingFace compatible format
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
    output_dir = run_dir / "vllm_mac_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting model from {checkpoint_file} to HuggingFace format")
    print(f"Output directory: {output_dir}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    
    # Extract model state dict
    state_dict = checkpoint['student_state_dict']
    
    # Save the transformed state dict
    torch.save(state_dict, output_dir / "pytorch_model.bin")
    
    # Try to find config file if not provided
    if config_path is None:
        # Look in checkpoint directory first
        checkpoint_dir = Path(checkpoint_file).parent
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
    parser.add_argument("--run", type=str, help="Run directory name (defaults to latest)")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file name (defaults to latest)")
    parser.add_argument("--config_path", type=str, help="Path to the model config file (defaults to initial_config.json in checkpoint dir)")
    
    args = parser.parse_args()
    
    # Convert paths if provided
    run_dir = Path("runs") / args.run if args.run else None
    
    export_model_to_hf_format(run_dir, args.checkpoint, args.config_path) 