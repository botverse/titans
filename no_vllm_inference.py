import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
import argparse

# Import your model classes
from models.llama_titans import MACTransformer, MACModule

def safe_logits_processor(logits):
    """Process logits safely to prevent numerical instability"""
    # Replace inf and -inf
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    
    # Clamp values
    logits = torch.clamp(logits, min=-1e4, max=1e4)
    
    # Apply softmax with better numerical stability
    logits = logits - logits.max(dim=-1, keepdim=True)[0]
    
    return logits

def run_inference(model_path, prompts, max_new_tokens=64, use_mac=False):
    """Run basic inference without vLLM"""
    # Load config
    model_path = Path(model_path)
    print(f"Loading model from {model_path}")
    
    if use_mac:
        # Load config for MAC model
        with open(model_path / "config.json", "r") as f:
            config_dict = json.load(f)
        config = LlamaConfig.from_dict(config_dict)
        
        # Initialize MAC module
        mac_module = MACModule(
            dim=config.hidden_size,
            **config.mac_module_config
        )
        
        # Initialize model with MAC
        model = MACTransformer(config=config, mac_module=mac_module)
        model = model.to(torch.float16)  # Use half precision
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # Use standard AutoModelForCausalLM loading
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use half precision for efficiency
            device_map="auto"  # Let HF determine optimal device placement
        )
    
    # Print model parameter count for reference
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # Move to GPU if available
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add model inspection before inference
    print("\nModel structure check:")
    for name, param in model.named_parameters():
        print(f"{name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    
    # Add generation config inspection
    generation_config = {
        "max_new_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "do_sample": True,
        "repetition_penalty": 1.1,
        "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True
    }
    print("\nGeneration config:", generation_config)
    
    results = []
    for prompt in prompts:
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)
        
        # Explicitly create attention mask to avoid ambiguity
        attention_mask = inputs['attention_mask']
        
        with torch.inference_mode():
            output_ids = model.generate(
                inputs.input_ids,
                attention_mask=attention_mask,  # explicitly pass attention mask
                max_new_tokens=generation_config["max_new_tokens"],
                do_sample=generation_config["do_sample"],
                temperature=generation_config["temperature"],
                top_p=generation_config["top_p"],
                top_k=generation_config["top_k"],
                repetition_penalty=generation_config["repetition_penalty"],
                pad_token_id=generation_config["pad_token_id"],
                eos_token_id=generation_config["eos_token_id"],
                use_cache=True
            )
            
            # Print intermediate logits for debugging
            with torch.no_grad():
                logits = model(inputs.input_ids).logits
                print(f"Output logits shape: {logits.shape}")
                print(f"Logits stats - mean: {logits.mean().item():.6f}, std: {logits.std().item():.6f}")
                
                # Check top predictions
                last_logits = logits[0, -1]
                top_tokens = torch.topk(last_logits, k=5)
                print("\nTop 5 predictions for next token:")
                for score, token_id in zip(top_tokens.values, top_tokens.indices):
                    token = tokenizer.convert_ids_to_tokens([token_id])[0]
                    print(f"Token: {token}, Score: {score.item():.6f}")
            
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"Prompt: {prompt}\nGenerated: {generated_text}\n")
            results.append((prompt, generated_text))
    
    return results

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run basic inference on distilled model")
    parser.add_argument("--use_mac", action="store_true", help="Use MAC-enhanced model")
    parser.add_argument("--model_path", type=str, help="Direct path to the model directory")
    parser.add_argument("--run", type=str, help="Run directory name (defaults to latest)")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    args = parser.parse_args()
    
    # Get the actual model path
    model_path = get_model_path(args)
    if not model_path.exists():
        raise ValueError(f"Model directory not found at {model_path}")
    
    prompts = [
        "what's the capital of france?",
        "what's the capital of germany?",
        "what's the capital of japan?"
    ]
    
    run_inference(model_path, prompts, args.max_tokens, args.use_mac)
