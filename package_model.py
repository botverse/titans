import os
import shutil
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer # Keep tokenizer loading
import argparse
import hashlib

# Assuming your custom code is in the 'models' directory relative to this script
MODELS_CODE_DIR = Path("models")
# Default HF cache structure relative to project root
DEFAULT_HF_CACHE = Path(".huggingface/hub")
DEFAULT_MODEL_NAME = "models--meta-llama--Meta-Llama-3-8B-Instruct"
DEFAULT_SNAPSHOT = "5f0b02c75b57c5855da9ae460ce51323ea669d8a"

def get_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def should_copy_file(src: Path, dest: Path) -> bool:
    """Check if file should be copied based on existence and hash comparison."""
    if not dest.exists():
        return True
    return get_file_hash(src) != get_file_hash(dest)

def package_base_model_with_wrapper(
    output_dir: str,
    base_model_snapshot_path: str = str(DEFAULT_HF_CACHE / DEFAULT_MODEL_NAME / "snapshots" / DEFAULT_SNAPSHOT), # Updated default
    # --- Wrapper Config ---
    memory_size: int = 4096,
    num_retrieved: int = 1,
    update_alpha: float = 0.1,
    surprise_momentum: float = 0.9,
    surprise_lr: float = 0.01,
):
    """
    Packages the base Llama model weights with the InferenceMemoryWrapper custom code.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model_path = Path(base_model_snapshot_path) # Convert to Path

    if not base_model_path.exists():
        # Try resolving symlinks in the path components if needed (e.g., if hub itself is a link)
        try:
            base_model_path = base_model_path.resolve(strict=True)
        except FileNotFoundError:
             raise FileNotFoundError(
                f"Base model snapshot path not found: {base_model_snapshot_path}. "
                f"Ensure the path exists or provide it using --base_model_snapshot_path."
            )
        if not base_model_path.exists():
             raise FileNotFoundError(
                 f"Resolved base model snapshot path not found: {base_model_path}."
             )


    print(f"Packaging base model from {base_model_path} with wrapper into {output_dir}")

    # --- 1. Copy tokenizer files ---
    print("Copying tokenizer files...")
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        # Add vocab.json or merges.txt if they exist and are needed
    ]
    for file_name in tokenizer_files:
        src = base_model_path / file_name
        if src.exists():
            # Resolve symlink and copy the actual file content
            try:
                shutil.copy2(src.resolve(strict=True), output_dir / file_name)
                print(f"  - Copied {file_name}")
            except FileNotFoundError:
                 print(f"Warning: Could not resolve symlink or copy file: {src}")
            except Exception as e:
                 print(f"Warning: Error copying {file_name}: {e}")
        else:
            print(f"Warning: Tokenizer file not found: {src}")

    # --- 2. Copy and modify config.json ---
    print("Processing config.json...")
    config_src = base_model_path / "config.json"
    if not config_src.exists():
        raise FileNotFoundError(f"Base model config.json not found at {config_src.resolve()}")

    try:
        with open(config_src.resolve(strict=True)) as f: # Resolve symlink
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not resolve symlink for config.json at {config_src}")
    except Exception as e:
        raise RuntimeError(f"Error reading config.json: {e}")


    # Modify config for your wrapper
    config["architectures"] = ["InferenceMemoryWrapper"] # Critical for AutoModel loading
    config["auto_map"] = { # Helps AutoModel find the custom class
        "AutoModelForCausalLM": "models.inference_memory_wrapper.InferenceMemoryWrapper"
    }
    # Add wrapper parameters (these will be passed to __init__)
    config["memory_size"] = memory_size
    config["num_retrieved"] = num_retrieved
    config["update_alpha"] = update_alpha
    config["surprise_momentum"] = surprise_momentum
    config["surprise_lr"] = surprise_lr
    # Add any other params your InferenceMemoryWrapper.__init__ takes

    # Ensure model type remains compatible if needed, or use a custom one
    # config["model_type"] = "llama" # Or maybe keep as is from base model

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- 3. Copy generation_config.json ---
    print("Copying generation_config.json...")
    gen_config_src = base_model_path / "generation_config.json"
    if gen_config_src.exists():
        try:
            shutil.copy2(gen_config_src.resolve(strict=True), output_dir / "generation_config.json") # Resolve symlink
        except FileNotFoundError:
             print(f"Warning: Could not resolve symlink for generation_config.json at {gen_config_src}")
        except Exception as e:
             print(f"Warning: Error copying generation_config.json: {e}")
    else:
        print(f"Warning: generation_config.json not found at {gen_config_src}")

    # --- 4. Copy custom model code ---
    print("Copying custom model code...")
    models_dest_dir = output_dir / "models"
    models_dest_dir.mkdir(exist_ok=True)

    # Create __init__.py
    with open(models_dest_dir / "__init__.py", "w") as f:
        f.write("# Required for Hugging Face to find custom code\n")
        f.write("from .inference_memory_wrapper import InferenceMemoryWrapper\n")
        # Add imports for other custom classes if they are needed by the wrapper

    # Copy necessary model files (adjust list as needed)
    required_code_files = ["inference_memory_wrapper.py"] # Add others like llama_titans.py if needed
    for file_name in required_code_files:
        src_file = MODELS_CODE_DIR / file_name
        if src_file.exists():
            shutil.copy2(src_file, models_dest_dir / file_name)
            print(f"  - Copied {file_name}")
        else:
            print(f"Warning: Custom code file not found: {src_file}")

    # --- 5. Copy base model weights (safetensors) ---
    print("Copying model weights...")
    # Find model files and index using glob and checking existence
    model_files = list(base_model_path.glob("model-*.safetensors"))
    index_file = base_model_path / "model.safetensors.index.json"

    if not model_files or not index_file.exists():
        # Check for pytorch_model.bin as fallback
        bin_file = base_model_path / "pytorch_model.bin"
        if bin_file.exists():
             print("Warning: Found pytorch_model.bin instead of safetensors. Copying .bin file.")
             model_files = [bin_file]
             index_file = None # No index for single bin file
        else:
            raise FileNotFoundError(
                f"Could not find model weight files (.safetensors and index.json, or pytorch_model.bin) in {base_model_path}"
            )

    # Copy weight files only if changed
    for file_path in model_files:
        dest_path = output_dir / file_path.name
        try:
            if should_copy_file(file_path.resolve(strict=True), dest_path):
                shutil.copy2(file_path.resolve(strict=True), dest_path)
                print(f"  - Copied {file_path.name} (file changed)")
            else:
                print(f"  - Skipped {file_path.name} (no changes)")
        except FileNotFoundError:
            print(f"Warning: Could not resolve symlink or copy weight file: {file_path}")
        except Exception as e:
            print(f"Warning: Error copying {file_path.name}: {e}")

    # Copy the index file if it exists (for safetensors) and has changed
    if index_file and index_file.exists():
        dest_index = output_dir / index_file.name
        try:
            if should_copy_file(index_file.resolve(strict=True), dest_index):
                shutil.copy2(index_file.resolve(strict=True), dest_index)
                print(f"  - Copied {index_file.name} (file changed)")
            else:
                print(f"  - Skipped {index_file.name} (no changes)")
        except FileNotFoundError:
             print(f"Warning: Could not resolve symlink for index file: {index_file}")
        except Exception as e:
            print(f"Warning: Error copying {index_file.name}: {e}")


    # --- 6. Create README.md (optional but recommended) ---
    print("Creating README.md...")
    readme_content = f"""
---
license: apache-2.0 # Or the appropriate license
library_name: transformers
tags:
- llama
- memory-augmented
---

# Memory-Augmented Llama Model ({config.get('_name_or_path', 'Llama-3-8B-Instruct')})

This repository contains the base weights for `{config.get('_name_or_path', 'Llama-3-8B-Instruct')}` packaged with custom code for the `InferenceMemoryWrapper`.

This allows loading the model with memory capabilities using `trust_remote_code=True`.

## Model Details
- **Base Model:** {config.get('_name_or_path', 'Llama-3-8B-Instruct')}
- **Wrapper:** `InferenceMemoryWrapper`
- **Memory Size:** {config.get('memory_size', 'N/A')}
- **Memory Dims:** {config.get('hidden_size', 'N/A')}
- **Memory Storage (approx):** {(config.get('memory_size', 0) * config.get('hidden_size', 0) * 2 * 2) / (1024*1024):.1f} MB (FP16, buffer + state) if buffer/state exist

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch # Added import for example

model_id = "your-username/your-repo-name" # Replace with your repo ID

# Load the model and tokenizer, allowing custom code execution
# Requires sufficient VRAM for the Llama 8B model + memory buffer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16, # Recommended for memory
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Example prompt
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate using the custom method
# Note: The memory buffer is initially randomly initialized unless loaded separately.
# It will be updated during generation if update_rule is 'ema' or 'surprise'.
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    use_memory=True,
    update_rule='ema' # or 'surprise' or 'none'
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

# To save user-specific memory state (after generation/updates):
# user_memory_state = model.memory_buffer.data.clone()
# user_surprise_state = model.surprise_state.clone()
# torch.save({{'memory_buffer': user_memory_state, 'surprise_state': user_surprise_state}}, 'user_memory.pt')

# To load user-specific memory state:
# loaded_state = torch.load('user_memory.pt')
# model.memory_buffer.data.copy_(loaded_state['memory_buffer'])
# model.surprise_state.copy_(loaded_state['surprise_state'])

```

**Important:** The `memory_buffer` and `surprise_state` in this packaged model are initialized randomly according to the `InferenceMemoryWrapper` code. They do **not** contain any pre-trained memory state unless you load it separately after initializing the model (see example above). You need to manage loading/saving the memory state per user externally.
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content.strip())

    print(f"\nModel packaged successfully in '{output_dir}'")
    print("This package includes the base Llama weights and the custom wrapper code.")
    print("Remember to push this directory to the Hugging Face Hub.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package base Llama model with InferenceMemoryWrapper code.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the packaged model."
    )
    # Let the function default handle the base model path unless overridden
    parser.add_argument(
        "--base_model_snapshot_path", type=str,
        # No default here, uses the function's default
        help=(
            "Path to the snapshot directory containing the base model weights "
            f"(defaults to relative path: {DEFAULT_HF_CACHE / DEFAULT_MODEL_NAME / 'snapshots' / DEFAULT_SNAPSHOT})"
        )
    )
    # Add arguments for wrapper parameters if you want them configurable via CLI
    parser.add_argument("--memory_size", type=int, default=4096)
    parser.add_argument("--num_retrieved", type=int, default=1)
    parser.add_argument("--update_alpha", type=float, default=0.1)
    parser.add_argument("--surprise_momentum", type=float, default=0.9)
    parser.add_argument("--surprise_lr", type=float, default=0.01)


    args = parser.parse_args()

    # Use the provided path or let the function use its default
    base_model_path_arg = args.base_model_snapshot_path if args.base_model_snapshot_path else str(DEFAULT_HF_CACHE / DEFAULT_MODEL_NAME / "snapshots" / DEFAULT_SNAPSHOT)


    package_base_model_with_wrapper(
        output_dir=args.output_dir,
        base_model_snapshot_path=base_model_path_arg, # Pass the determined path
        memory_size=args.memory_size,
        num_retrieved=args.num_retrieved,
        update_alpha=args.update_alpha,
        surprise_momentum=args.surprise_momentum,
        surprise_lr=args.surprise_lr,
    )

    print("\nNext steps:")
    print(f"1. Navigate to '{args.output_dir}'")
    print("2. Initialize git: `git init && git lfs install`")
    print("3. Track large files: `git lfs track \"*.safetensors\" \"*.bin\"`") # Track both
    print("4. Add files: `git add .`")
    print("5. Commit: `git commit -m 'Package base Llama with InferenceMemoryWrapper'`")
    print("6. Create a repository on Hugging Face Hub.")
    print("7. Add remote: `git remote add origin https://huggingface.co/your-username/your-repo-name`")
    print("8. Push: `git push -u origin main`")