import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaConfig
import os
from pathlib import Path
import sys
import argparse # Added for command-line arguments

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

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Test Llama model with InferenceMemoryWrapper.")
parser.add_argument(
    '--test-mem',
    action='store_true',
    help='Run the fictitious country memory test (Experiment 1).'
)
parser.add_argument(
    '--prompt',
    type=str,
    default="What is the capital of France?",
    help='Default prompt for basic generation test.'
)
parser.add_argument(
    '--max-new-tokens',
    type=int,
    default=50,
    help='Maximum new tokens for standard generation test.'
)
parser.add_argument(
    '--use-memory',
    action=argparse.BooleanOptionalAction, # Allows --use-memory / --no-use-memory
    default=True,
    help='Enable memory usage during standard generation.'
)
parser.add_argument(
    '--update-rule',
    type=str,
    default='ema',
    choices=['ema', 'surprise', 'none'],
    help='Memory update rule to use for standard generation.'
)
parser.add_argument(
    '--model-dir',
    type=str,
    default="../titans-hf/",
    help='Path to the packaged model directory.'
)
parser.add_argument(
    '--no-quant',
    action='store_true',
    help='Disable 4-bit quantization.'
)
parser.add_argument(
    '--test-context-window',
    type=int,
    default=1024, # Add argument for test window size
    help='Artificial context window size for memory experiment.'
)

args = parser.parse_args()


# --- Configuration ---
MODEL_DIR = args.model_dir # Use path from args

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        f"Model directory not found: {MODEL_DIR}. "
        "Make sure this path points to the directory created by the packaging script."
    )

# --- Quantization Configuration ---
if args.no_quant:
    quantization_config = None
    print("Quantization: DISABLED by command line flag.")
else:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # Or torch.float16
        bnb_4bit_use_double_quant=True,
    )
    print(f"Quantization: 4-bit (NF4)")


print("--- Loading Model and Tokenizer ---")
print(f"Model directory: {MODEL_DIR}")
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
        # Ensure loaded parameter data is explicitly assigned and cast
        wrapper.memory_buffer.data = loaded_param.data.to(target_device).to(expected_dtype) # (M, C) Note: Shape depends on saved tensor
        print(f"  - Loaded memory_buffer.pt to {wrapper.memory_buffer.device}, dtype {wrapper.memory_buffer.dtype}")
    else:
        print("  - Warning: memory_buffer.pt not found. Using initial wrapper values.")
        # Move the existing buffer (already correct dtype) to the target device
        wrapper.memory_buffer.data = wrapper.memory_buffer.data.to(target_device)


    if surprise_state_path.exists():
        # Load potentially float32 tensor, move to target device, and cast to expected dtype
        surprise_state_loaded = torch.load(surprise_state_path, map_location=target_device)
        # Cast loaded tensor to the expected dtype
        surprise_state_casted = surprise_state_loaded.to(expected_dtype)
        # Manually assign to the registered buffer
        # Correct assignment for buffers
        with torch.no_grad():
             wrapper.surprise_state.copy_(surprise_state_casted)

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
    model = wrapper # The wrapper now contains the potentially quantized base_model
    # No need to move wrapper again, device_map handled base model placement,
    # and we moved buffer/state manually. Wrapper attributes point to base model parts.
    print(f"\nWrapper manually loaded. Type: {type(model)}, Buffer Device: {model.memory_buffer.device}")

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


# === Standard Generation Test (always run) ===
print("\n\n=== Starting Standard Generation Test ===")

print("\n--- Preparing Input ---")
print(f"Prompt: {args.prompt}")
try:
    inputs = tokenizer(args.prompt, return_tensors="pt")
    # Determine input device
    input_device = model.get_input_embeddings().weight.device
    inputs = inputs.to(input_device)
    print(f"Input tensor device: {inputs.input_ids.device}")
except Exception as e:
    print(f"Error tokenizing input: {e}")
    exit(1)

print("\n--- Generating Text ---")
print(f"Using memory: {args.use_memory}, Update rule: {args.update_rule}")

try:
    # No need for autocast here if base model loaded with torch_dtype=float16 or bnb compute_dtype
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            use_memory=args.use_memory,
            update_rule=args.update_rule,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n--- Standard Output ---")
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
    print(f"\n--- An error occurred during standard generation ---")
    print(e)
    import traceback
    traceback.print_exc()


# === Experiment 1: Fictitious Country Memory Test (conditional) ===
if args.test_mem:
    print("\n\n=== Starting Experiment 1: Fictitious Country Memory ===")
    print(f"Using artificial context window: {args.test_context_window}")

    # --- Define Data ---
    country_name = "Eldoria"
    capital_name = "Lumina"
    memorable_data = f"The Kingdom of {country_name} is a land of ancient forests and sparkling rivers. Its capital city is {capital_name}."
    # --- CORRECTED filler_data ---
    filler_data = (
        f" The capital sits nestled in the Azure Valley, famed for its bioluminescent flora that light up the nights. "
        f"The kingdom's economy relies heavily on the export of Sunstone crystals, mined from the northern peaks. "
        f"The people there are known for their intricate woodwork and harmonious music. "
        f"The ruling monarch, Queen Aris, resides in the Crystal Palace in the capital city. "
        f"The city's Grand Library holds scrolls detailing the kingdom's long history, including the legendary tale of the Starfall Dragon. "
        f"The local cuisine often features river salmon and wild berries, prepared with glowing moonpetal herbs. "
        f"The climate is temperate, with warm, shimmering summers and mild, snowy winters where the Frostbloom flowers appear. "
        f"The Azure River flows through the capital, powering the city's ingenious water clocks and reflecting the sky gardens above. "
        f"Trade routes connect the capital to the coastal city of Starfall and the mountain citadel of Ironpeak. "
        f"Festivals in the kingdom, like the Sunstone Jubilee and the Nightbloom Gala, are celebrated with elaborate light displays. "
        f"The architecture in the capital blends natural elements with crystalline structures, creating a unique aesthetic. "
        f"Scholars in the capital study the ancient celestial alignments said to influence the land's magic. "
        f"Travelers speak of the Whispering Woods near the kingdom's border, where the trees are said to hold memories. "
        f"The national symbol of the kingdom is the Azure Phoenix, often depicted soaring over the capital city. "
        f"Guards in the capital wear armor polished with Sunstone dust, giving it a faint glow. "
        f"The city market is a bustling place filled with exotic goods from across the kingdom and beyond. "
        f"Children in the kingdom learn the traditional art of Sky-weaving, creating tapestries that shimmer with captured light. "
        f"The capital's observatory is renowned for its powerful crystal lenses, allowing astronomers to map distant galaxies. "
        # Add more filler to ensure context length > test_context_window
        * 5 # Repeat filler to make it longer
    )
    question = f"\n\nQuestion: What is the capital city of the Kingdom of {country_name}?"

    full_prompt = memorable_data + filler_data + question
    max_new_tokens_exp1 = 15 # Keep generation short for the answer

    # --- Prepare Full and Truncated Inputs ---
    print("\n--- Preparing Inputs for Experiment 1 ---")
    try:
        full_inputs = tokenizer(full_prompt, return_tensors="pt")
        full_input_ids = full_inputs.input_ids
        full_length = full_input_ids.shape[1]

        if full_length <= args.test_context_window:
             print(f"\nWarning: Full prompt length ({full_length}) is not greater than the test context window ({args.test_context_window}).")
             print("The experiment might not effectively test memory beyond the direct context.")

        # Truncate to the last 'test_context_window' tokens
        truncated_input_ids = full_input_ids[:, -args.test_context_window:]
        truncated_attention_mask = torch.ones_like(truncated_input_ids) # Mask for truncated input

        # Move tensors to the correct device
        input_device_exp1 = model.get_input_embeddings().weight.device
        full_input_ids = full_input_ids.to(input_device_exp1)
        truncated_input_ids = truncated_input_ids.to(input_device_exp1)
        truncated_attention_mask = truncated_attention_mask.to(input_device_exp1)

        print(f"Full Prompt Length: {full_length} tokens")
        print(f"Truncated Input Length: {truncated_input_ids.shape[1]} tokens")
        print(f"Input tensor device: {input_device_exp1}")

    except Exception as e:
        print(f"Error tokenizing/preparing input for Experiment 1: {e}")
        exit(1)

    # --- Run Generation (With Memory) ---
    print("\n--- Generating Text (With Memory) ---")
    print("  Step 1: Priming memory with full context (max_new_tokens=0)...")
    answer_with_memory = "[ERROR DURING GENERATION]"
    try:
        # 1a. Prime Memory (process full context with memory updates enabled)
        with torch.no_grad(): # No gradients needed for priming with EMA
            _ = model.generate(
                input_ids=full_input_ids,
                max_new_tokens=0, # Just process, don't generate
                use_memory=True,
                update_rule='ema', # Update memory using EMA rule
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        print("  Memory primed.")

        # 1b. Generate Answer (process truncated context with memory retrieval only)
        print("  Step 2: Generating answer from truncated context (retrieval only)...")
        with torch.no_grad():
             outputs_with_memory = model.generate(
                 input_ids=truncated_input_ids,
                 attention_mask=truncated_attention_mask, # Provide mask for truncated input
                 max_new_tokens=max_new_tokens_exp1,
                 use_memory=True,        # Retrieve from primed memory
                 update_rule='none',     # Do not update memory during final answer generation
                 do_sample=False,
                 temperature=0.1,
                 top_p=1.0,
                 pad_token_id=tokenizer.pad_token_id,
                 eos_token_id=tokenizer.eos_token_id,
             )
        # Decode the full output from the generation step
        generated_text_with_memory_full = tokenizer.decode(outputs_with_memory[0], skip_special_tokens=True)
        # Extract the answer part *after* the truncated prompt text
        truncated_prompt_text = tokenizer.decode(truncated_input_ids[0], skip_special_tokens=True)
        answer_with_memory = generated_text_with_memory_full.replace(truncated_prompt_text, "").strip()

    except Exception as e:
        print(f"\n--- An error occurred during generation (With Memory) ---")
        print(e)
        import traceback
        traceback.print_exc()
        answer_with_memory = "[ERROR DURING GENERATION]"

    # --- Run Generation (Without Memory) ---
    print("\n--- Generating Text (Without Memory) ---")
    print("  Generating answer from truncated context only...")
    answer_without_memory = "[ERROR DURING GENERATION]"
    try:
        # Generate directly from truncated input, memory disabled
        with torch.no_grad():
            outputs_without_memory = model.generate(
                input_ids=truncated_input_ids,
                attention_mask=truncated_attention_mask, # Provide mask for truncated input
                max_new_tokens=max_new_tokens_exp1,
                use_memory=False,       # Memory disabled
                update_rule='none',     # No update
                do_sample=False,
                temperature=0.1,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Decode the full output
        generated_text_without_memory_full = tokenizer.decode(outputs_without_memory[0], skip_special_tokens=True)
        # Extract the answer part *after* the truncated prompt text
        truncated_prompt_text = tokenizer.decode(truncated_input_ids[0], skip_special_tokens=True)
        answer_without_memory = generated_text_without_memory_full.replace(truncated_prompt_text, "").strip()

    except Exception as e:
        print(f"\n--- An error occurred during generation (Without Memory) ---")
        print(e)
        import traceback
        traceback.print_exc()
        answer_without_memory = "[ERROR DURING GENERATION]"


    # --- Compare Results ---
    print("\n\n--- Experiment 1 Results ---")
    print(f"(Context Window: {args.test_context_window}, Full Prompt Length: {full_length})")
    print(f"Expected Answer: {capital_name}")
    print("-" * 30)
    print(f"Answer WITH Memory (Primed Full Context, Generated Truncated): '{answer_with_memory}'")
    print(f"Answer WITHOUT Memory (Generated Truncated Only):             '{answer_without_memory}'")
    print("-" * 30)

    # Check if the exact capital name is present (case-insensitive)
    # Check if the capital name starts the answer string (more specific)
    with_memory_correct = answer_with_memory.lower().strip().startswith(capital_name.lower())
    without_memory_correct = answer_without_memory.lower().strip().startswith(capital_name.lower())


    if with_memory_correct:
        print("✅ Model WITH memory correctly recalled the capital.")
    else:
        print("❌ Model WITH memory FAILED to recall the capital.")

    if without_memory_correct:
        print("✅ Model WITHOUT memory correctly recalled the capital.")
    else:
        print("❌ Model WITHOUT memory FAILED to recall the capital.")

# End of conditional experiment block
else:
    print("\nSkipping Experiment 1: Fictitious Country Memory Test (run with --test-mem to enable).")


print("\n--- Test Complete ---")