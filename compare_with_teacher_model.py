import dotenv
dotenv.load_dotenv()
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', device_map='auto')
input_ids = tokenizer('what is the capital of france?', return_tensors='pt').input_ids.to('cuda')
outputs = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
