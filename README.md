```sh
# Distillation
poetry run python distil.py

# Convert to vLLM format
poetry run python export_model.py --checkpoint checkpoints/checkpoint_epoch_1.pt --output_dir vllm_mac_model

# Run inference
poetry run python vllm_inference.py --use_mac --model_path vllm_mac_model --prompt "Hello, how are you?"





# Create a vLLM model without the mac module
poetry run python prepare_for_vllm.py --checkpoint checkpoints/checkpoint_epoch_1.pt --output_dir vllm_model_no_mac
```