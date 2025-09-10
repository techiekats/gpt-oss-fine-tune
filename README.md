# gpt-oss-fine-tune

## Based on these instructions: https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers 
## docker compose up

Go to the terminal of the container.

1. Install the model and run the fine tune scripts
## python fine-tune.py


2. Run the inference and generate some tokens

## python inference.py

NOTE: the finetuning failed with the error: Your setup doesn't support bf16/gpu.