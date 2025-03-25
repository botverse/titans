from vllm import LLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # e.g., "meta-llama/Meta-Llama-3-8B-Instruct"
llm = LLM(
    model=model_id,
    dtype=torch.float16,
    trust_remote_code=True,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    max_model_len=2048  # ✅ Correct argument
)

# Generate text
prompt = "Hello, my name is John Doe, what is my name?"
response = llm.generate(prompt)
print(response)

# Export the model
def export_model(llm, export_path):
    # Hypothetical way to access the model's state_dict
    # This might be different based on the actual API
    model_state_dict = llm.model.state_dict()  # Replace with actual method to get state_dict

    # Save the model's state_dict
    torch.save(model_state_dict, export_path)

# Specify the export path
export_path = "quantized_model.pth"
export_model(llm, export_path)

print(f"Model exported to {export_path}")
