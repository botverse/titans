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
        
        with torch.no_grad():
            # Generate text
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.3,  # Lower temperature for more focused output
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode the generated text
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
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
