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

def main():
    parser = argparse.ArgumentParser(description="Run inference with vLLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--prompt", type=str, default="What is machine learning?", help="Prompt for inference")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--use_mac", action="store_true", help="Use MAC-enhanced model")
    parser.add_argument("--force_fp16", action="store_true", help="Force FP16 precision")
    args = parser.parse_args()
    
    # Set up vLLM sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
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
    outputs = llm.generate(args.prompt, sampling_params)
    
    # Print the generated text
    for output in outputs:
        print(f"Generated text: {output.outputs[0].text}")

def run_mac_inference(args, sampling_params):
    """Run inference with MAC model using custom implementation"""
    # We need to register our custom model with the transformers library
    # Ensure the models directory is in the path
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    
    # Load model path and copy necessary files
    model_path = Path(args.model_path)
    
    # Register our custom model class
    
    # Register the custom model type
    class MACLlamaConfig(LlamaConfig):
        model_type = "llama_mac"
    
    CONFIG_MAPPING.register("llama_mac", MACLlamaConfig)
    
    # Now load the config and model
    config = AutoConfig.from_pretrained(model_path)
    
    # Manually create the model with our custom class
    model = MACLlamaForCausalLM.from_pretrained(
        model_path, 
        config=config,
        torch_dtype=torch.float16 if args.force_fp16 else None
    )
    model.to("cuda")
    
    # Load tokenizer directly
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Generate text using our custom model's generation method
    print(f"Input prompt: {args.prompt}")
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to("cuda")
    
    # Using manual generation with our MAC model
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            use_mac=True  # Enable MAC functionality
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main() 