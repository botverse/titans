import os
import argparse
import torch
import sys
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from models.llama_titans import MACTransformer, MACModule
from transformers import LlamaConfig
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from models.vllm_mac_model import MACLlamaForCausalLM
import json

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

def get_model_path(args):
    """Get model path from arguments, handling both direct paths and run directories"""
    if args.run:
        # Use specified run directory
        run_dir = Path("runs") / args.run
    else:
        # Find latest run
        run_dir = find_latest_run()
        if run_dir is None:
            # If no run found and model_path specified, use it directly
            if args.model_path:
                return Path(args.model_path)
            raise ValueError("No experiment runs found in runs directory")
    
    # Determine model directory name based on MAC usage
    model_dir = "vllm_mac_model" if args.use_mac else "vllm_llama_model"
    return run_dir / model_dir

def main():
    parser = argparse.ArgumentParser(description="Run inference with vLLM")
    parser.add_argument("--model_path", type=str, help="Direct path to the model directory")
    parser.add_argument("--run", type=str, help="Run directory name (defaults to latest)")
    parser.add_argument("--prompt", type=str, default="What is machine learning?", help="Prompt for inference")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--use_mac", action="store_true", help="Use MAC-enhanced model")
    parser.add_argument("--force_fp16", action="store_true", help="Force FP16 precision", default=True)
    parser.add_argument(
        "--memory_efficient", 
        action="store_true",
        default=True,
        help="Enable memory efficient settings"
    )
    args = parser.parse_args()
    
    # Get the actual model path
    model_path = get_model_path(args)
    if not model_path.exists():
        raise ValueError(f"Model directory not found at {model_path}")
    
    # Update args.model_path for the rest of the code
    args.model_path = str(model_path)
    
    # Set up vLLM sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    if args.memory_efficient:
        # Set PyTorch memory allocator settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:True'
        torch.cuda.empty_cache()
    
    if args.use_mac:
        print("Using MAC-enhanced model (custom implementation)")
        run_mac_inference(args, sampling_params)
    else:
        print("Using standard vLLM model (without memory)")
        run_standard_inference(args, sampling_params)

def run_standard_inference(args, sampling_params):
    """Run inference using standard vLLM without MAC"""
    # Load the model through vLLM
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="half" if args.force_fp16 else "auto",
        trust_remote_code=True
    )
    
    # Run inference
    print(f"Input prompt: {args.prompt}")
    torch.cuda.empty_cache()
    outputs = llm.generate(args.prompt, sampling_params)
    
    # Print the generated text
    for output in outputs:
        print(f"Generated text: {output.outputs[0].text}")

def run_mac_inference(args, sampling_params):
    """Run inference with MAC model using custom implementation"""
    # Clear CUDA cache and set memory-efficient settings
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32,expandable_segments:True'
    
    # Load model config
    model_path = Path(args.model_path)
    with open(model_path / "config.json", "r") as f:
        config_dict = json.load(f)
    
    # Optimize config for inference
    config = LlamaConfig.from_dict(config_dict)
    config.use_memory_efficient_attention = True
    config.attention_implementation = "eager"
    
    # Initialize model
    model = MACLlamaForCausalLM(config)
    
    # Load state dict with proper key mapping
    state_dict = torch.load(
        model_path / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True  # Important: reduce memory usage during loading
    )
    
    # Fix state dict keys - this is the critical fix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.llama.'):
            # Remove both 'model.' and 'llama.' prefixes
            new_key = k.replace('model.llama.', '')
            new_state_dict[f'model.{new_key}'] = v
        elif k.startswith('llama.'):
            # Remove 'llama.' prefix
            new_key = k.replace('llama.', '')
            new_state_dict[f'model.{new_key}'] = v
        elif k.startswith('model.'):
            # Keep as is
            new_state_dict[k] = v
        else:
            # Add 'model.' prefix
            new_state_dict[f'model.{k}'] = v
    
    # Load state dict
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing or unexpected:
        print(f"Missing keys: {missing[:5]}...")
        print(f"Unexpected keys: {unexpected[:5]}...")
    
    # Move to GPU with memory optimizations
    if args.force_fp16:
        model = model.half()
    model = model.to("cuda", non_blocking=True)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Process input with reduced sequence length
    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128  # Reduce from 256 to save memory
    ).to("cuda", non_blocking=True)
    
    print(f"Input prompt: {args.prompt}")
    
    # Generate with memory optimizations
    try:
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=min(args.max_tokens, 64),  # Limit output length
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                use_mac=True,
                use_cache=True,  # Enable KV caching
                repetition_penalty=1.1
            )
        
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nOOM error. Current memory usage:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
            print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
            print("\nTry reducing sequence length or model size")
        raise e

if __name__ == "__main__":
    main() 