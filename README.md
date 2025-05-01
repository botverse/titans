# Titans Paper Implementation

** ⚠️ Disclaimer: Experimental ⚠️ **

**This repository is an *experimental* and *unofficial* implementation of ideas from the [TITANS paper](https://arxiv.org/abs/2402.19429) and related memory-augmented transformer research.**
- It is **not** an official release, nor is it guaranteed to match the results or architecture of the original paper.
- The code is for research, exploration, and educational purposes only. Use at your own risk!

---

## Project Overview

This project explores memory-augmented transformer architectures inspired by the TITANS paper, focusing on:
- **Memory as a Context (MAC):** Integrating persistent and long-term memory into Llama models.
- **Surprise-based Memory Updates:** Implementing gradient-based "surprise" learning rules for memory.
- **Distillation:** Training smaller student models from a large Llama teacher, optionally with memory modules.
- **vLLM Compatibility:** Exporting models for efficient inference with [vLLM](https://github.com/vllm-project/vllm).
- **Custom Inference Wrappers:** Wrapping Llama models with memory buffers for plug-and-play memory-augmented inference.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Implemented Architectures](#implemented-architectures)
- [Quickstart](#quickstart)
- [Tool and Script Guide](#tool-and-script-guide)
- [Memory-Augmented Model Usage](#memory-augmented-model-usage)
- [Development Utilities](#development-utilities)
- [Citation](#citation)
- [License](#license)

---

## Implemented Architectures

- **MACTransformer / MACLlamaForCausalLM:**  
  Implements "Memory as a Context" (MAC) as described in the TITANS paper, with persistent and long-term memory buffers, and both EMA and surprise-based update rules.  
  See: `models/llama_titans.py`, `models/vllm_mac_model.py`

- **InferenceMemoryWrapper:**  
  A HuggingFace-compatible wrapper for Llama models, adding a memory buffer and surprise/EMA update logic for inference.  
  See: `models/inference_memory_wrapper.py`

- **Naive TitansModel:**  
  A minimal, didactic implementation of a memory-augmented transformer block for educational purposes.  
  See: `models/naive_model.py`

- **PaperMemory:**  
  A direct, minimal translation of the memory update equations from the TITANS paper, for reference and testing.  
  See: `models/paper_memory.py`

---

## Quickstart

### 1. **Distill a Student Model (with or without MAC)**
```sh
poetry run python distil.py --use-mac  # Add --use-mac to enable memory module
```

### 2. **Export for vLLM or HuggingFace Inference**
- **With MAC (custom memory):**
  ```sh
  poetry run python export_mac_model.py --run <run_name> --checkpoint <checkpoint_file>
  ```
- **Standard Llama (no MAC):**
  ```sh
  poetry run python export_llama_model.py --run <run_name> --checkpoint <checkpoint_file>
  ```

### 3. **Run Inference**
- **With vLLM (standard or MAC):**
  ```sh
  poetry run python vllm_inference.py --use_mac --model_path vllm_mac_model --prompt "Hello, how are you?"
  ```
- **Without vLLM (direct PyTorch):**
  ```sh
  poetry run python no_vllm_inference.py --use_mac --model_path vllm_mac_model
  ```

### 4. **Test the InferenceMemoryWrapper**
```sh
poetry run python testmodel.py --model-dir <path_to_packaged_model> --prompt "What is the capital of France?" --use-memory --update-rule ema
```

---

## Tool and Script Guide

### **Training & Distillation**
- `distil.py`  
  Main script for distilling a student model from a Llama teacher. Supports both standard and MAC-augmented students.  
  **Key args:** `--use-mac`, `--batch-size`, `--max-length`, `--num-epochs`, `--resume`

- `train.py`  
  Minimal trainer for the naive TitansModel on OpenWebText (for educational/ablation purposes).

### **Export & Packaging**
- `export_mac_model.py`  
  Exports a MAC-augmented model to HuggingFace format, including custom code and config.

- `export_llama_model.py`  
  Converts a distilled model to standard Llama format for vLLM or HuggingFace inference.

- `package_model.py`  
  Packages a base Llama model with the `InferenceMemoryWrapper` for plug-and-play memory-augmented inference.

### **Inference**
- `vllm_inference.py`  
  Runs inference using vLLM, supporting both standard and MAC-augmented models.

- `no_vllm_inference.py`  
  Runs inference directly with PyTorch/HuggingFace, supporting both standard and MAC models.

- `testmodel.py`  
  Tests the `InferenceMemoryWrapper` with various prompts and memory update rules. Includes a "fictitious country" memory recall experiment.

### **Analysis & Utilities**
- `inspect_training_examples.py`  
  Prints random or sequential samples from the training dataset.

- `inspect_state_dict.py`, `inspect_weights.py`, `verify_export_process.py`, `verify_model_architecture.py`  
  Scripts for inspecting, verifying, and debugging model checkpoints and exports.

- `check_tokenizer_alignment.py`, `compare_with_teacher_model.py`  
  Tools for verifying tokenizer and output alignment between teacher and student models.

- `quantize.py`  
  Example script for quantizing a Llama model with bitsandbytes.

- `bin/`  
  Contains bash scripts for distributed training, inference, and notifications.

---

## Memory-Augmented Model Usage

### **Loading a Packaged Memory Model**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "your-username/your-repo-name"  # Replace with your repo ID

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    use_memory=True,
    update_rule='ema'  # or 'surprise' or 'none'
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Saving/Loading Memory State:**
```python
# Save
torch.save({'memory_buffer': model.memory_buffer.data, 'surprise_state': model.surprise_state}, 'user_memory.pt')
# Load
state = torch.load('user_memory.pt')
model.memory_buffer.data.copy_(state['memory_buffer'])
model.surprise_state.copy_(state['surprise_state'])
```

---

## Development Utilities

- **Tokenizer/Model Alignment:**  
  Use `check_tokenizer_alignment.py` and `compare_with_teacher_model.py` to ensure your exported models and tokenizers match the originals.

- **Model Inspection:**  
  Use `inspect_state_dict.py`, `inspect_weights.py`, and `verify_export_process.py` to debug and verify model weights and architecture.

- **Dataset Inspection:**  
  Use `inspect_training_examples.py` to view and debug training data.

---

## Citation

If you use this codebase for research, please cite the original TITANS paper:

```
@article{titans2024,
  title={TITANS: Memory-Augmented Transformers with Surprise-based Learning},
  author={...},
  journal={arXiv preprint arXiv:2402.19429},
  year={2024}
}
```

---

## License

This code is provided under the Apache 2.0 License (see LICENSE file).  
**Note:** The base Llama weights and tokenizer are subject to their own licenses and terms of use.

---

**Again: This is an experimental, unofficial implementation for research and educational purposes only.**
