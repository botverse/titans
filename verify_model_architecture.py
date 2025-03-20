import torch
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM

def verify_model(model_path):
    """Verify model architecture and weights"""
    model_path = Path(model_path)
    
    # Load config
    config = AutoConfig.from_pretrained(model_path)
    print("\nModel config:")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num attention heads: {config.num_attention_heads}")
    print(f"Num hidden layers: {config.num_hidden_layers}")
    print(f"Vocab size: {config.vocab_size}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Check embedding dimensions
    embed = model.get_input_embeddings()
    print("\nEmbedding check:")
    print(f"Embedding shape: {embed.weight.shape}")
    print(f"Matches vocab size: {embed.weight.shape[0] == config.vocab_size}")
    print(f"Matches hidden size: {embed.weight.shape[1] == config.hidden_size}")
    
    # Check lm_head dimensions
    lm_head = model.get_output_embeddings()
    print("\nLM head check:")
    print(f"LM head shape: {lm_head.weight.shape}")
    print(f"Matches vocab size: {lm_head.weight.shape[0] == config.vocab_size}")
    print(f"Matches hidden size: {lm_head.weight.shape[1] == config.hidden_size}")
    
    # Verify weight sharing if applicable
    print("\nWeight sharing check:")
    if hasattr(model, "tie_weights"):
        print("Model has tie_weights capability")
        is_tied = torch.equal(embed.weight, lm_head.weight)
        print(f"Weights are tied: {is_tied}")
    
    return model

if __name__ == "__main__":
    latest_run = Path("runs").glob("distil_*").__next__()
    model_path = latest_run / "vllm_llama_model"
    verify_model(model_path) 