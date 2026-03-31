from huggingface_hub import login
from utils.config import BASE_DIR, hf_token
from transformers import AutoTokenizer, AutoModelForCausalLM

login(token = hf_token)

model_name = "gpt2"
model_id = f"openai-community/{model_name}"
save_directory = f"{BASE_DIR}/model/{model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")