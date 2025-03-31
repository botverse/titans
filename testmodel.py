import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaConfig
import os
from pathlib import Path
import sys

# --- Add models directory to Python path ---
# Assuming testmodel.py is in titans/ and models/ is also in titans/
SCRIPT_DIR = Path(__file__).parent.resolve()
MODELS_DIR_PATH = SCRIPT_DIR / 'models'
if str(MODELS_DIR_PATH) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR_PATH))

# --- Now import the custom class ---
try:
    from inference_memory_wrapper import InferenceMemoryWrapper
except ImportError as e:
    print(f"Error importing InferenceMemoryWrapper: {e}")
    print(f"Ensure 'models/inference_memory_wrapper.py' exists and is in the Python path.")
    print(f"Attempted to add '{MODELS_DIR_PATH}' to path.")
    exit(1)


# --- Configuration ---
MODEL_DIR = "../titans-hf/" # Adjust this path if needed

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        f"Model directory not found: {MODEL_DIR}. "
        "Make sure this path points to the directory created by the packaging script."
    )

prompt = "What is the capital of France?"
max_new_tokens = 50
use_memory_flag = True
memory_update_rule = 'ema'

# --- Quantization Configuration (Re-enabled) ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Or torch.float16
    bnb_4bit_use_double_quant=True,
)
# quantization_config = None # Keep disabled if testing without quantization

print("--- Loading Model and Tokenizer ---")
print(f"Model directory: {MODEL_DIR}")
print(f"Quantization: {'4-bit (NF4)' if quantization_config else 'DISABLED'}")
print(f"Device Mapping: auto")

# --- Load Tokenizer ---
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        print("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)

# --- Manual Loading Strategy ---
try:
    print("Loading BASE Llama model...")
    # 1. Load the base Llama model normally using from_pretrained
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=quantization_config, # Apply quantization here
        device_map="auto",                       # Apply device map here
        trust_remote_code=False, # Don't load custom code for the base model
        low_cpu_mem_usage=True,
        # Let quantization config handle dtype, or use float16 if no quantization
        torch_dtype=torch.float16 if quantization_config is None else None
    )
    base_model.eval() # Ensure base model is in eval mode
    expected_dtype = base_model.dtype # Get the actual dtype base model is using
    print(f"Base model loaded. Type: {type(base_model)}, Device: {base_model.device}, Dtype: {expected_dtype}")


    print("\nInstantiating InferenceMemoryWrapper...")
    # 2. Instantiate the wrapper, passing the loaded base model
    config = LlamaConfig.from_pretrained(MODEL_DIR)

    wrapper = InferenceMemoryWrapper(
         llama_model=base_model, # Pass the loaded base model instance
         memory_size=getattr(config, "memory_size", 4096),
         num_retrieved=getattr(config, "num_retrieved", 1),
         update_alpha=getattr(config, "update_alpha", 0.1),
         surprise_momentum=getattr(config, "surprise_momentum", 0.9),
         surprise_lr=getattr(config, "surprise_lr", 0.01),
    )
    # Wrapper state (memory buffer, surprise state) is initialized with correct dtype but on CPU.

    print("\nLoading wrapper state (memory buffer, surprise state)...")
    # 3. Manually load the wrapper's state
    # Determine a representative device from the base model if using device_map="auto"
    try:
        # Accessing .device might fail if model is entirely on meta device initially with device_map
        target_device = base_model.device
    except AttributeError:
         # Fallback if .device isn't directly available (might happen with complex device_map setups)
         # Get device of the first parameter we can find
         target_device = next(base_model.parameters()).device
         print(f"Using device of first base model parameter: {target_device}")


    memory_buffer_path = Path(MODEL_DIR) / "memory_buffer.pt"
    surprise_state_path = Path(MODEL_DIR) / "surprise_state.pt"

    if memory_buffer_path.exists():
        # Load the saved parameter (potentially float32), move & cast
        # Use map_location to load directly to the target device if possible
        loaded_param = torch.load(memory_buffer_path, map_location=target_device)
        # Detach data, cast to expected dtype, assign back to wrapper's parameter data
        wrapper.memory_buffer.data = loaded_param.data.to(expected_dtype) # {{ edit }} Load param directly, cast data
        print(f"  - Loaded memory_buffer.pt to {wrapper.memory_buffer.device}, dtype {wrapper.memory_buffer.dtype}")
    else:
        print("  - Warning: memory_buffer.pt not found. Using initial wrapper values.")
        # Move the existing buffer (already correct dtype) to the target device
        wrapper.memory_buffer.data = wrapper.memory_buffer.data.to(target_device)


    if surprise_state_path.exists():
        # Load potentially float32 tensor, move to target device, and cast to expected dtype
        surprise_state_loaded = torch.load(surprise_state_path, map_location=target_device)
        # Cast loaded tensor to the expected dtype (float16)
        surprise_state_casted = surprise_state_loaded.to(expected_dtype)
        # Manually assign to the registered buffer
        wrapper.surprise_state = surprise_state_casted
        print(f"  - Loaded surprise_state.pt to {wrapper.surprise_state.device}, dtype {wrapper.surprise_state.dtype}")
    else:
        print("  - Warning: surprise_state.pt not found. Using initial wrapper values.")
        # Move the existing buffer (already correct dtype) to the target device
        wrapper.surprise_state = wrapper.surprise_state.to(target_device)

    # Ensure the wrapper is also in eval mode (though llama part is already)
    wrapper.eval()
    # Ensure the wrapper itself is placed correctly if using device_map="auto"
    # We need the memory buffer on the correct device for the matmul
    # The base_model might be split, but the matmul happens where the query is.
    # Let's explicitly move the wrapper (including its Parameter/Buffers) to the target device.
    model = wrapper.to(target_device) # {{ edit }} Move the whole wrapper to ensure buffer is moved
    print(f"\nWrapper manually loaded and moved. Type: {type(model)}, Buffer Device: {model.memory_buffer.device}")

except ValueError as ve:
     # Catch the specific error if it still somehow occurs during base model loading
     if "too many values to unpack" in str(ve):
         print(f"ERROR: The 'ValueError: too many values to unpack' occurred even during BASE model loading.")
         print("This is unexpected and might indicate an issue with the base model files or HF cache.")
     else:
         print(f"An unexpected ValueError occurred during loading: {ve}")
     exit(1)
except Exception as e:
    print(f"Error during manual loading process: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# --- Prepare Input ---
# (Input preparation remains the same, but ensure it goes to the correct device)
print("\n--- Preparing Input ---")
print(f"Prompt: {prompt}")
try:
    inputs = tokenizer(prompt, return_tensors="pt")
    # Determine input device - should match the embedding layer's device
    # Note: with device_map='auto', embeddings might be on CPU or GPU
    input_device = model.get_input_embeddings().weight.device
    inputs = inputs.to(input_device)
    print(f"Input tensor device: {inputs.input_ids.device}")
except Exception as e:
    print(f"Error tokenizing input: {e}")
    exit(1)

# --- Run Generation ---
# (Generation logic remains the same, using the manually loaded 'model' object)
print("\n--- Generating Text ---")
print(f"Using memory: {use_memory_flag}, Update rule: {memory_update_rule}")

try:
    # No need for autocast here if base model loaded with torch_dtype=float16 or bnb compute_dtype
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_memory=use_memory_flag,
            update_rule=memory_update_rule,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n--- Output ---")
    print(generated_text)

except torch.cuda.OutOfMemoryError:
    print("\n--- CUDA Out of Memory Error ---")
    print("Ran out of VRAM. Check if quantization is active and memory usage.")
    try:
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
    except:
        pass # Might fail if CUDA context is already lost
except Exception as e:
    print(f"\n--- An error occurred during generation ---")
    print(e)
    import traceback
    traceback.print_exc()


print("\n--- Test Complete ---")