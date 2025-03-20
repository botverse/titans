import dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

dotenv.load_dotenv()

def find_latest_run():
    runs_dir = Path("runs")
    experiments = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("distil_")]
    return max(experiments, key=lambda x: x.stat().st_mtime) if experiments else None

# Create offload directory
offload_dir = Path("model_offload")
offload_dir.mkdir(exist_ok=True)

# Load teacher model with proper offloading
teacher_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
teacher_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Meta-Llama-3-8B-Instruct',
    device_map='auto',
    offload_folder=str(offload_dir)
)

# Load student model
latest_run = find_latest_run()
student_path = latest_run / "vllm_llama_model"
student_tokenizer = AutoTokenizer.from_pretrained(str(student_path))
student_model = AutoModelForCausalLM.from_pretrained(
    str(student_path),
    device_map='auto',
    offload_folder=str(offload_dir),
    torch_dtype=torch.float16  # Use half precision to save memory
)

# Test prompt
prompt = "what is the capital of france?"
input_ids_teacher = teacher_tokenizer(prompt, return_tensors='pt').input_ids.to('cuda')
input_ids_student = student_tokenizer(prompt, return_tensors='pt').input_ids.to('cuda')

# Generate from both models
with torch.inference_mode():
    teacher_outputs = teacher_model.generate(
        input_ids_teacher,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True
    )
    student_outputs = student_model.generate(
        input_ids_student,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True
    )

print("Teacher output:", teacher_tokenizer.decode(teacher_outputs[0], skip_special_tokens=True))
print("Student output:", student_tokenizer.decode(student_outputs[0], skip_special_tokens=True))
