from huggingface_hub import notebook_login
from datasets import load_dataset


notebook_login()

dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")

