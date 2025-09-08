from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, Mxfp4Config
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
from trl import SFTTrainer


notebook_login()

dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

messages = dataset[0]["messages"]
conversation = tokenizer.apply_chat_template(messages, tokenize=False)

quantization_config = Mxfp4Config(dequantize=True)
model_kwargs = dict(
    attn_implementation="eager", # for better performance
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False, #since we will fine-tune the model with gradient checkpointing
    device_map="auto",
    #local_files_only=True
)
## download model
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **model_kwargs)

## PEFT (Parameter efficient fine-tuning)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",
    target_parameters=[
        "7.mlp.experts.gate_up_proj",
        "7.mlp.experts.down_proj",
        "15.mlp.experts.gate_up_proj",
        "15.mlp.experts.down_proj",
        "23.mlp.experts.gate_up_proj",
        "23.mlp.experts.down_proj",
    ],
)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

training_args = SFTConfig(
    learning_rate=2e-4,
    gradient_checkpointing=True,
    num_train_epochs=1,
    logging_steps=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_length=2048,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    output_dir="gpt-oss-20b-multilingual-reasoner",
    report_to="trackio",
    push_to_hub=True,
)

trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()

trainer.save_model(training_args.output_dir)
trainer.push_to_hub(dataset_name="HuggingFaceH4/Multilingual-Thinking")
print ('pushed to hub')