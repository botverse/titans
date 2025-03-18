import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
import argparse

# Import your model classes
from models.llama_titans import MACTransformer, MACModule

def run_inference(model_path, prompts, max_new_tokens=100, use_mac=False):
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = []
    for prompt in prompts:
        # Tokenize
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"  # Use proper Llama3 format
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256  # Much longer than 8!
        ).to(device)
        
        # Generate with detailed tensor shape logging
        print(f"\nProcessing prompt: {prompt}")
        print(f"Input shape: {inputs.input_ids.shape}")  # (B, T)
        
        with torch.inference_mode():
            if use_mac:
                # Generate text with MAC model
                output_ids = model.generate(
                    tokens=inputs.input_ids,  # MAC model expects 'tokens'
                    max_new_tokens=64,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_mac=True  # Enable MAC functionality
                )
            else:
                # Standard generation
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"Generated: {generated_text}")
            results.append((prompt, generated_text))
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run basic inference on distilled model")
    parser.add_argument("--use_mac", action="store_true", help="Use MAC-enhanced model")
    parser.add_argument("--model_path", type=str, help="Path to the model directory")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    args = parser.parse_args()
    
    # Set default model path based on --use_mac if not explicitly provided
    if args.model_path is None:
        args.model_path = "vllm_mac_model" if args.use_mac else "vllm_model"
    
    prompts = [
        "what's the capital of france?",
        "what's the capital of germany?",
        "what's the capital of japan?"
    ]
    
    run_inference(args.model_path, prompts, args.max_tokens, args.use_mac)
