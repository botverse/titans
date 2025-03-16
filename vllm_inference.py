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

def main():
    parser = argparse.ArgumentParser(description="Run inference with vLLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
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
    
    # Optimize config for inference without changing dimensions
    config = LlamaConfig.from_dict(config_dict)
    config.use_memory_efficient_attention = True
    config.attention_implementation = "eager"
    
    # Memory optimizations
    config.max_position_embeddings = min(config.max_position_embeddings, 1024)  # Limit context
    config.use_cache = True  # Enable KV caching
    
    # Initialize model
    model = MACLlamaForCausalLM(config)
    model.eval()  # Ensure eval mode
    
    # Load state dict efficiently
    state_dict_path = model_path / "pytorch_model.bin"
    if state_dict_path.exists():
        # Load with reduced memory usage
        state_dict = torch.load(
            state_dict_path,
            map_location="cpu",
            weights_only=True
        )
        
        # Fix state dict keys
        new_state_dict = {
            k.replace('llama.model.', 'model.'): v 
            for k, v in state_dict.items()
        }
        
        # Convert to FP16 if requested
        if args.force_fp16:
            new_state_dict = {k: v.half() for k, v in new_state_dict.items()}
        
        model.load_state_dict(new_state_dict, strict=False)
    
    # Move to GPU efficiently
    if args.force_fp16:
        model = model.half()
    model = model.to("cuda", non_blocking=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Process input with memory constraints
    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256  # Reasonable context length
    )
    
    inputs = {k: v.to("cuda", non_blocking=True) for k, v in inputs.items()}
    
    # Memory-efficient generation settings
    gen_kwargs = {
        "max_new_tokens": min(args.max_tokens, 128),  # Limit output length
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
        "use_mac": True,
        "use_cache": True,
        "repetition_penalty": 1.1
    }
    
    # Generate with memory optimizations
    try:
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            output_ids = model.generate(**{**inputs, **gen_kwargs})
            
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nGenerated text: {generated_text}")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nOOM error. Memory usage:")
            print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
            print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
            print(f"Free: {torch.cuda.memory_reserved() - torch.cuda.memory_allocated() / 1e9:.2f}GB")
            print("\nTry:")
            print("1. Reducing max_tokens")
            print("2. Reducing input sequence length")
            print("3. Using a smaller model")
        raise

if __name__ == "__main__":
    main() 