from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

# Load the original model first
model_kwargs = dict(attn_implementation="eager", torch_dtype="auto", use_cache=True, device_map="auto")
base_model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **model_kwargs).cuda()

# Merge fine-tuned weights with the base model
peft_model_id = "gpt-oss-20b-multilingual-reasoner"
model = PeftModel.from_pretrained(base_model, peft_model_id)
model = model.merge_and_unload()

## Test case 1:
REASONING_LANGUAGE = "German"
SYSTEM_PROMPT = f"reasoning language: {REASONING_LANGUAGE}"
USER_PROMPT = "¿Cuál es el capital de Australia?"  # Spanish for "What is the capital of Australia?"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

gen_kwargs = {"max_new_tokens": 512, "do_sample": True, "temperature": 0.6, "top_p": None, "top_k": None}

output_ids = model.generate(input_ids, **gen_kwargs)
response = tokenizer.batch_decode(output_ids)[0]
print(response)


## Test case 2:
REASONING_LANGUAGE = "Chinese"  # or Hindi, or any other language...
SYSTEM_PROMPT = f"reasoning language: {REASONING_LANGUAGE}"
USER_PROMPT = "What is the national symbol of Canada?"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

output_ids = model.generate(input_ids, **gen_kwargs)
response = tokenizer.batch_decode(output_ids)[0]
print(response)