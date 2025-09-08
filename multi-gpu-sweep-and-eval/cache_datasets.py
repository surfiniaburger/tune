
"""
==============================================================================
CACHE_DATASETS.PY (v9 - Conditional Labeling for Training vs. Eval)
==============================================================================
This script now uses a Python generator with `Dataset.from_generator` and
integrates a robust `albumentations` pipeline for the training set.

v9 FIXES:
- Solves the `NaN` training loss by excluding the `labels` column for the
  training set, forcing the trainer to use standard next-token prediction.
- Conditionally INCLUDES the `labels` column ONLY for the validation set,
  ensuring that the evaluation script has the ground truth data it needs.
"""

import os
import shutil
import argparse
import numpy as np
import albumentations as A
from datasets import Dataset
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

# --- 1. Configuration ---
MODEL_NAME = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"
LOCAL_DATA_PATH = Path("/kaggle/working/local_datasets/")

SOURCE_DATA_PATHS = { "train": "/kaggle/input/maize-dataset/", "validation": "/kaggle/input/aura-mind-maize-validation/", }
LOCAL_PATHS = { "train": LOCAL_DATA_PATH / "train", "validation": LOCAL_DATA_PATH / "validation", }
CACHED_PATHS = { "train": "data/cached_train_dataset_hf_augmented", "validation": "data/cached_val_dataset_hf", }
CLASS_NAME_MAPPING = {
    "maize_healthy": "Healthy Maize Plant",
    "phosphorus_deficiency": "Maize Phosphorus Deficiency",
}
CLASS_ID_MAPPING = {
    "maize_healthy": 0,
    "phosphorus_deficiency": 1,
}



# --- 2. Data Copying Function ---
def copy_data_if_needed(dataset_type: str):
    source_path, local_path = SOURCE_DATA_PATHS[dataset_type], LOCAL_PATHS[dataset_type]
    print(f"--- Checking and Copying Data for: {dataset_type} ---")
    if not local_path.exists():
        print(f"Copying data from {source_path} to {local_path}..."); shutil.copytree(source_path, local_path); print(f"✅ Data copy complete for {dataset_type}.")
    else: print(f"✅ {dataset_type.capitalize()} data already exists at {local_path}")

# --- 3. Generator-Based Data Processing ---
def create_conversation_dict(image_path, class_name):
    display_name = CLASS_NAME_MAPPING.get(class_name, "Unknown Maize Condition")
    try:
        return { "messages": [ { "role": "user", "content": [ {"type": "text", "text": "Classify the condition of this maize plant. Choose from: Healthy Maize Plant, Maize Phosphorus Deficiency."}, {"type": "image", "image": Image.open(image_path).convert("RGB")} ] }, { "role": "assistant", "content": [ {"type": "text", "text": f"This is a {display_name}."} ] }, ] }
    except Exception as e: print(f"Warning: Could not process image {image_path}. Error: {e}"); return None

# MODIFIED: Added `dataset_type` to conditionally add labels
def sample_generator(image_paths, processor, dataset_type, augmentations=None):
    for path in tqdm(image_paths, desc="Processing samples"):
        class_name = path.parent.name
        raw_sample = create_conversation_dict(path, class_name)
        if raw_sample is None:
            continue

        pil_image = raw_sample["messages"][0]['content'][1]['image']
        image_to_process = np.array(pil_image)
        if augmentations:
            image_to_process = augmentations(image=image_to_process)['image']

        text = processor.tokenizer.apply_chat_template(
            raw_sample["messages"], tokenize=False, add_generation_prompt=False
        )
        text_inputs = processor.tokenizer(text, padding=False, truncation=True, return_tensors="pt")
        image_inputs = processor.image_processor(images=[image_to_process], return_tensors="pt")

        # Base sample dictionary without labels
        sample_output = {
            "input_ids": text_inputs.input_ids.squeeze(),
            "attention_mask": text_inputs.attention_mask.squeeze(),
            "pixel_values": image_inputs.pixel_values.squeeze(),
        }
        
        # *** THE FIX ***
        # Only add the `labels` column for the validation set.
        # This prevents the trainer from seeing it and causing a NaN loss,
        # but allows the evaluation script to use it for scoring.
        if dataset_type == 'validation':
            label_id = CLASS_ID_MAPPING.get(class_name, -1)
            sample_output["labels"] = [label_id]

        yield sample_output


def process_and_cache_dataset(dataset_type: str):
    image_dir, cache_path = LOCAL_PATHS[dataset_type], CACHED_PATHS[dataset_type]
    if os.path.exists(cache_path):
        print(f"✅ Dataset '{dataset_type}' already cached at {cache_path}. Skipping."); return
    print(f"\n--- Processing dataset: {dataset_type} from: {image_dir} ---")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    image_paths = list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.jpeg"))
    print(f"Found {len(image_paths)} images.")
    augmentations = None
    if dataset_type == 'train':
        print("Defining ROBUST augmentation pipeline for training data...")
        # Restored augmentations
        augmentations = A.Compose([
            A.HorizontalFlip(p=0.5), A.Rotate(limit=25, p=0.4),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-20, 20), p=0.3),
            A.GaussianBlur(p=0.3), A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.2), A.ISONoise(intensity=(0.1, 0.5), p=0.1),
            A.Sharpen(p=0.2), A.CLAHE(p=0.3), A.Equalize(p=0.2),
            A.ColorJitter(p=0.4), A.Posterize(p=0.1), A.Solarize(p=0.1),
            A.CoarseDropout(p=0.2),
        ])
        print("✅ Augmentation pipeline created successfully.")
        
    print("Creating dataset from generator...")
    # MODIFIED: Pass `dataset_type` to the generator's arguments
    processed_dataset = Dataset.from_generator(
        sample_generator,
        gen_kwargs={
            "image_paths": image_paths,
            "processor": processor,
            "dataset_type": dataset_type, # <-- Pass the context here
            "augmentations": augmentations,
        }
    )
    os.makedirs(cache_path, exist_ok=True)
    print(f"Saving processed dataset to directory: {cache_path}")
    processed_dataset.save_to_disk(cache_path)
    print(f"\n✅ Dataset '{dataset_type}' successfully processed and saved to '{cache_path}'.")

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache a specific dataset."); parser.add_argument("--dataset-type", type=str, required=True, choices=['train', 'validation']); args = parser.parse_args()
    LOCAL_DATA_PATH.mkdir(exist_ok=True); copy_data_if_needed(args.dataset_type); process_and_cache_dataset(args.dataset_type)
    print(f"\n--- Caching process for {args.dataset_type} is complete! ---")