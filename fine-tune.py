from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoTokenizer

notebook_login()

dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

messages = dataset[0]["messages"]
conversation = tokenizer.apply_chat_template(messages, tokenize=False)
print(conversation)

