# https://www.youtube.com/watch?v=-Fcb7OT-uC
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("HUGGINGFACE_KEY")

print(api_key)

# Set Hugging Face API Key from Environment Variable
huggingface_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

required_files = [
  "generation_config.json",
  "tokenizer_config.json",
  "model.safetensors",
  "tokenizer.json",
  "config.json"
]

for filename in required_files:
  download_location = hf_hub_download(
    repo_id=huggingface_model,
    filename=filename,
    token=api_key
  )
  print(f"File downloaded to: {download_location}")

model = AutoModelForCausalLM.from_pretrained(huggingface_model)
tokenizer = AutoTokenizer.from_pretrained(huggingface_model)

text_generation_pipeline = pipeline(
  "text-generation",
  model=model,
  tokenizer=tokenizer,
  max_length=200,
  truncation=True,
)

response = text_generation_pipeline("Why did the chicken cross the road?")
print(response)


response2 = text_generation_pipeline("What lays hidden beneath the Earth?")
print(response2)
