import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
import argparse

# Import your model classes
from models.llama_titans import MACTransformer, MACModule

def run_inference(model_path, prompts, max_new_tokens=100):
    """Run basic inference without vLLM"""
    # Load config
    model_path = Path(model_path)
    print(f"Loading model from {model_path}")
    
    # Load model and tokenizer using HF's auto classes
    # This handles both .bin and .safetensors formats automatically
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
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate with detailed tensor shape logging
        print(f"\nProcessing prompt: {prompt}")
        print(f"Input shape: {inputs.input_ids.shape}")  # (B, T)
        
        with torch.no_grad():
            # Generate text
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Decode the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated: {generated_text}")
            results.append((prompt, generated_text))
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run basic inference on distilled model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    args = parser.parse_args()
    
    prompts = [
        "what's the capital of france?",
        "what's the capital of germany?",
        "what's the capital of japan?"
    ]
    
    run_inference(args.model_path, prompts, args.max_tokens)
