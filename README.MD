Step 1: Set up the Environment

You're already an expert at this. It's best to create a clean environment.

bash
 Show full code block 
# Create and activate a new virtual environment
python3 -m venv vlm-finetune-env
source vlm-finetune-env/bin/activate

# Install the mlx-vlm library
pip install mlx-vlm
Step 2: Prepare Your Dataset

This is the most important part. The mlx-vlm fine-tuning script expects a directory containing train.jsonl and valid.jsonl files. Each line in these files must be a JSON object with an "image" key (the path to the image) and a "text" key (the full conversational turn).

For example, your train.jsonl would look like this:

json
{"image": "dataset/images/product_A.jpg", "text": "[INST] Describe this image. [/INST] A vibrant red t-shirt with a minimalist logo, made from soft, breathable cotton."}
{"image": "dataset/images/product_B.jpg", "text": "[INST] What is this? [/INST] Sleek, wireless over-ear headphones in matte black, featuring noise-cancellation."}
Step 3: Run the Fine-Tuning Command

Instead of using the lora.py script from your text-only project, you invoke the lora module directly from the mlx-vlm library.

bash

python -m mlx_vlm.lora \
  --model-path mlx-community/gemma-3n-E2B-it-4bit \
  --dataset ./mlx_dataset/ \
  --apply-chat-template \
  --steps 100 \
  --lora-rank 16 \
  --batch-size 1 \
  --learning-rate 2e-5 \
  --print-every 10 \
  --output-path ./lora_adapters/

This command will:

Load the base VLM.
Read your jsonl data, automatically processing both the images and the text prompts.
Apply LoRA adapters to the model.
Run the training loop and save an adapters.npz file.
This is the direct MLX equivalent to the workflow described in the Google tutorial.

source .venv/bin/activate

python prepare_dataset.py