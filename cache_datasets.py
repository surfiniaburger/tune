"""
==============================================================================
CACHE_DATASETS.PY (v4 - With Albumentations)
==============================================================================
This script now uses a Python generator with `Dataset.from_generator` and
integrates a robust `albumentations` pipeline for the training set.

Usage remains the same:
- python cache_datasets.py --dataset-type train
- python cache_datasets.py --dataset-type validation
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
import numpy as np
import albumentations as A

# --- 1. Configuration ---
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

# --- 2. Data Copying Function ---
def copy_data_if_needed(dataset_type: str):
    source_path = SOURCE_DATA_PATHS[dataset_type]
    local_path = LOCAL_PATHS[dataset_type]
    print(f"--- Checking and Copying Data for: {dataset_type} ---")
    if not local_path.exists():
        print(f"Copying data from {source_path} to {local_path}...")
        shutil.copytree(source_path, local_path)
        print(f"✅ Data copy complete for {dataset_type}.")
    else:
        print(f"✅ {dataset_type.capitalize()} data already exists at {local_path}")

# --- 3. Generator-Based Data Processing with Augmentations ---

def create_conversation_dict(image_path, class_name):
    display_name = CLASS_NAME_MAPPING.get(class_name, "Unknown Maize Condition")
    try:
        # Keep as PIL.Image for now; conversion to NumPy happens in the generator
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

def sample_generator(image_paths, processor, augmentations=None):
    """
    A generator that processes and yields one sample at a time,
    applying augmentations if provided.
    """
    print("Starting sample generator...")
    for path in tqdm(image_paths, desc="Processing samples"):
        raw_sample = create_conversation_dict(path, path.parent.name)
        if raw_sample is None:
            continue

        pil_image = raw_sample["messages"][0]['content'][1]['image']
        image_to_process = pil_image

        # Apply augmentations if a pipeline is provided (for training set)
        if augmentations:
            image_np = np.array(pil_image)
            augmented = augmentations(image=image_np)
            image_to_process = augmented['image'] # This is now a NumPy array

        # Apply chat template to get the text part
        text = processor.tokenizer.apply_chat_template(
            raw_sample["messages"], tokenize=False, add_generation_prompt=False
        )

        # Process text and image (image_processor handles both PIL and NumPy)
        text_inputs = processor.tokenizer(text, padding=False, truncation=True, return_tensors="pt")
        image_inputs = processor.image_processor(images=[image_to_process], return_tensors="pt")

        yield {
            "input_ids": text_inputs.input_ids.squeeze(),
            "attention_mask": text_inputs.attention_mask.squeeze(),
            "pixel_values": image_inputs.pixel_values.squeeze(),
        }

def process_and_cache_dataset(dataset_type: str):
    image_dir = LOCAL_PATHS[dataset_type]
    cache_path = CACHED_PATHS[dataset_type]

    if os.path.exists(cache_path):
        print(f"✅ Dataset '{dataset_type}' already cached at {cache_path}. Skipping.")
        return

    print(f"\n--- Processing dataset: {dataset_type} from: {image_dir} ---")
    print(f"Loading processor for '{MODEL_NAME}'...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    image_paths = list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.jpeg"))
    print(f"Found {len(image_paths)} images.")

    augmentations = None
    if dataset_type == 'train':
        print("Defining ROBUST augmentation pipeline for training data...")
        # Using the user-provided albumentations pipeline, but EXCLUDING Normalize,
        # as the model's processor handles that.
        augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=25, p=0.4),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.MotionBlur(blur_limit=(3, 7), p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.ISONoise(intensity_limit=(0.1, 0.5), p=0.1),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Equalize(p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            A.Posterize(num_bits=4, p=0.1),
            A.Solarize(threshold=128, p=0.1),
            A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, p=0.2),
            # NOTE: A.Resize and A.Normalize are deliberately excluded.
            # The model's processor will handle resizing and normalization.
        ])

    print("Creating dataset from generator...")
    processed_dataset = Dataset.from_generator(
        sample_generator,
        gen_kwargs={"image_paths": image_paths, "processor": processor, "augmentations": augmentations},
    )

    os.makedirs(cache_path, exist_ok=True)
    print(f"Saving processed dataset to directory: {cache_path}")
    processed_dataset.save_to_disk(cache_path)
    print(f"\n✅ Dataset '{dataset_type}' successfully processed and saved to '{cache_path}'.")

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache a specific dataset (train or validation).")
    parser.add_argument("--dataset-type", type=str, required=True, choices=['train', 'validation'],
                        help="The type of dataset to process.")
    args = parser.parse_args()

    LOCAL_DATA_PATH.mkdir(exist_ok=True)
    copy_data_if_needed(args.dataset_type)
    process_and_cache_dataset(args.dataset_type)
    print(f"\n--- Caching process for {args.dataset_type} is complete! ---")
