# train.py (v2.1 - Corrected Batch Indexing)
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from mlx_vlm.utils import load
from src.aura_mind.data_utils import augment_image

# --- Configuration ---
MODEL_PATH = "mlx-community/gemma-3n-E2B-it-4bit"
DATA_PATH = "mlx_dataset"
ADAPTER_PATH = "lora_adapters"
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 1024

# --- Data Loading and Preparation ---
# In train.py

def data_iterator(dataset_path: str, processor, batch_size: int, is_training: bool):
    with open(dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    while True:
        indices = np.random.permutation(len(dataset))
        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_data = [dataset[idx] for idx in batch_indices]
            
            images, prompts = [], []
            for item in batch_data:
                try:
                    img = Image.open(item["image"]).convert("RGB")
                    if is_training:
                        img = augment_image(img)
                    images.append(img)
                    # We still need the raw text prompt for the processor
                    prompts.append(item["text"])
                except Exception as e:
                    print(f"Warning: could not process item {item['image']}. Error: {e}")
                    continue

            if not images:
                continue

            # --- THE FIX IS HERE ---
            # Call the processor ONCE with both images and text.
            # This is the expected usage and will prevent the error.
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            )
            # Convert to mlx tensors and yield the whole dictionary
            yield {k: mx.array(v) for k, v in inputs.items()}

# --- Training Loop ---
def main():
    print("🚀 Starting custom training loop (v2.1)...")

    model, processor = load(MODEL_PATH)
    model.unfreeze()

    def loss_fn(model, batch):
    # The batch from our new iterator already contains everything the model needs.
    # The processor automatically creates the 'labels' key from 'input_ids'.
        outputs = model(**batch)
        return outputs.loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE)

    train_iter = data_iterator(f"{DATA_PATH}/train.jsonl", processor, batch_size=1, is_training=True)
    
    print("🔥 Starting Training...")
    try:
        num_batches = len(list(open(f"{DATA_PATH}/train.jsonl")))
        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0
            with tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}") as pbar:
                for i, batch in zip(pbar, train_iter):
                    if not batch: continue
                    loss, grads = loss_and_grad_fn(model, batch)
                    optimizer.update(model, grads)
                    mx.eval(model.parameters(), optimizer.state)
                    
                    current_loss = loss.item()
                    epoch_loss += current_loss
                    pbar.set_postfix({"loss": f"{current_loss:.3f}"})
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

    print("✅ Training complete!")

if __name__ == "__main__":
    main()