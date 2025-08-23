"""
==============================================================================
CACHE_DATASETS.PY (v3 - Generator Based)
==============================================================================
This script now uses a Python generator with `Dataset.from_generator` to create
the cached dataset. This is the most memory-efficient method and avoids loading
the entire dataset into RAM at once, preventing OOM (Out of Memory) errors.

Usage remains the same:
- python cache_datasets.py --dataset-type train
- python cache_datasets.py --dataset-type validation
"""

import os
import shutil
import argparse
from datasets import Dataset
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

# --- 1. Configuration (Same as before) ---
MODEL_NAME = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"
LOCAL_DATA_PATH = Path("/kaggle/working/local_datasets/")

SOURCE_DATA_PATHS = {
    "train": "/kaggle/input/maize-dataset/",
    "validation": "/kaggle/input/aura-mind-maize-validation/",
}
LOCAL_PATHS = {
    "train": LOCAL_DATA_PATH / "train",
    "validation": LOCAL_DATA_PATH / "validation",
}
CACHED_PATHS = {
    "train": "data/cached_train_dataset_hf",
    "validation": "data/cached_val_dataset_hf",
}
CLASS_NAME_MAPPING = {
    "maize_healthy": "Healthy Maize Plant",
    "phosphorus_deficiency": "Maize Phosphorus Deficiency",
}

# --- 2. Data Copying Function (Same as before) ---
def copy_data_if_needed(dataset_type: str):
    source_path = SOURCE_DATA_PATHS[dataset_type]
    local_path = LOCAL_PATHS[dataset_type]
    print(f"--- Checking and Copying Data for: {dataset_type} ---")
    if not local_path.exists():
        print(f"Copying data from {source_path} to {local_path}...")
        shutil.copytree(source_path, local_path)
        print(f"âœ… Data copy complete for {dataset_type}.")
    else:
        print(f"âœ… {dataset_type.capitalize()} data already exists at {local_path}")

# --- 3. THE NEW GENERATOR-BASED APPROACH ---

def create_conversation_dict(image_path, class_name):
    display_name = CLASS_NAME_MAPPING.get(class_name, "Unknown Maize Condition")
    try:
        pil_image = Image.open(image_path).convert("RGB")
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Classify the condition of this maize plant. Choose from: Healthy Maize Plant, Maize Phosphorus Deficiency."},
                        {"type": "image", "image": pil_image}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"This is a {display_name}."}
                    ]
                },
            ]
        }
    except Exception as e:
        print(f"Warning: Could not process image {image_path}. Error: {e}")
        return None

def sample_generator(image_paths, processor):
    """A generator that processes and yields one sample at a time."""
    print("Starting sample generator...")
    for path in tqdm(image_paths, desc="Processing samples"):
        # Create the raw message dictionary
        raw_sample = create_conversation_dict(path, path.parent.name)
        if raw_sample is None:
            continue

        # Apply chat template to get the text part
        text = processor.tokenizer.apply_chat_template(
            raw_sample["messages"], tokenize=False, add_generation_prompt=False
        )
        # Get the image part
        image = raw_sample["messages"][0]['content'][1]['image']

        # Process text and image separately
        text_inputs = processor.tokenizer(text, padding=False, truncation=True, return_tensors="pt")
        image_inputs = processor.image_processor(images=[image], return_tensors="pt")

        # Yield the final processed dictionary
        yield {
            "input_ids": text_inputs.input_ids.squeeze(),
            "attention_mask": text_inputs.attention_mask.squeeze(),
            "pixel_values": image_inputs.pixel_values.squeeze(),
        }

def process_and_cache_dataset(dataset_type: str):
    image_dir = LOCAL_PATHS[dataset_type]
    cache_path = CACHED_PATHS[dataset_type]

    if os.path.exists(cache_path):
        print(f"âœ… Dataset '{dataset_type}' already cached at {cache_path}. Skipping.")
        return

    print(f"\n--- Processing dataset: {dataset_type} from: {image_dir} ---")
    print(f"Loading processor for '{MODEL_NAME}'...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    image_paths = list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.jpeg"))
    print(f"Found {len(image_paths)} images.")

    # Create the dataset using the generator
    print("Creating dataset from generator...")
    processed_dataset = Dataset.from_generator(
        sample_generator,
        gen_kwargs={"image_paths": image_paths, "processor": processor},
    )

    os.makedirs(cache_path, exist_ok=True)
    print(f"Saving processed dataset to directory: {cache_path}")
    processed_dataset.save_to_disk(cache_path)
    print(f"\nâœ… Dataset '{dataset_type}' successfully processed and saved to '{cache_path}'.")

# --- 4. Main Execution Block (Same as before) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache a specific dataset (train or validation).")
    parser.add_argument("--dataset-type", type=str, required=True, choices=['train', 'validation'],
                        help="The type of dataset to process.")
    args = parser.parse_args()

    LOCAL_DATA_PATH.mkdir(exist_ok=True)
    copy_data_if_needed(args.dataset_type)
    process_and_cache_dataset(args.dataset_type)
    print(f"\n--- Caching process for {args.dataset_type} is complete! ---")