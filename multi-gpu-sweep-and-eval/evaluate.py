"""
==============================================================================
EVALUATE.PY (v5 - Correct Multimodal Input Processing)
==============================================================================
This script evaluates a trained model adapter on the validation dataset.

This version is rectified to handle the specific input requirements of the
Gemma3N model. It reloads the raw images during evaluation to allow the
processor to correctly match the number of image placeholder tokens to the
number of image features, resolving the `ValueError`.
"""

import os
import json
import torch
from datasets import load_from_disk
from unsloth import FastVisionModel
from transformers import AutoProcessor
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# --- 1. Configuration ---
BASE_MODEL_NAME = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"
CACHED_VAL_PATH = "data/cached_val_dataset_hf"
CONFIG_PATH = "current_run_config.json"

# Path to the original validation images
# This must match the path used in your cache_datasets.py script
SOURCE_VAL_PATH = Path("/kaggle/working/local_datasets/validation/")

# Mapping from numeric ID to the keyword for scoring.
LABEL_ID_TO_KEYWORD = {
    0: "healthy",
    1: "phosphorus",
}


# --- 2. Helper Function ---
def is_prediction_correct(prediction: str, ground_truth_id: int) -> bool:
    pred_lower = prediction.lower()
    target_keyword = LABEL_ID_TO_KEYWORD.get(ground_truth_id)
    if not target_keyword:
        print(f"Warning: Unknown ground_truth_id: {ground_truth_id}")
        return False
    return "maize" in pred_lower and target_keyword in pred_lower

# --- 3. Main Evaluation Function ---
def main():
    print("-- Starting Evaluation --")

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Could not find {CONFIG_PATH}.")
    with open(CONFIG_PATH, 'r') as f:
        run_config = json.load(f)
        run_name = run_config["run_name"]
    
    relative_adapter_path = f"outputs/{run_name}/checkpoint-22"
    adapter_path = os.path.abspath(relative_adapter_path)
    
    print(f"Evaluating adapter: {adapter_path}")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter directory not found at: {adapter_path}")

    print(f"Loading base model: {BASE_MODEL_NAME}")
    model, processor = FastVisionModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    FastVisionModel.for_inference(model)
    print(f"Loading adapter from: {adapter_path}")
    model.load_adapter(adapter_path)

    print(f"Loading cached validation set from {CACHED_VAL_PATH}")
    validation_dataset = load_from_disk(CACHED_VAL_PATH)
    
    # --- START OF FIX ---
    # We must reload the raw image paths to process them correctly with the text.
    print(f"Loading raw image paths from {SOURCE_VAL_PATH}")
    image_paths = sorted(list(SOURCE_VAL_PATH.glob("**/*.jpg")) + list(SOURCE_VAL_PATH.glob("**/*.jpeg")))
    if len(image_paths) != len(validation_dataset):
        raise ValueError(f"Mismatch between number of images found ({len(image_paths)}) and size of cached dataset ({len(validation_dataset)}).")
    # --- END OF FIX ---

    correct_predictions = 0
    total_predictions = len(validation_dataset)

    # Use enumerate to iterate through both the cached data and the image paths
    for i, item in enumerate(tqdm(validation_dataset, desc="Evaluating")):
        ground_truth_label_id = item['labels'][0]
        
        # Load the corresponding raw image
        image_path = image_paths[i]
        raw_image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify the condition of this maize plant. Choose from: Healthy Maize Plant, Maize Phosphorus Deficiency."},
                    {"type": "image", "image": raw_image}, # <-- Pass the actual image here
                ],
            }
        ]
        
        # The processor now handles text and image together, creating the correct input format
        text_prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=text_prompt, images=raw_image, return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=20, use_cache=True)
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        prompt_marker = "model\n"
        answer_start_index = response.rfind(prompt_marker)
        final_answer = response[answer_start_index + len(prompt_marker):].strip() if answer_start_index != -1 else ""

        if is_prediction_correct(final_answer, ground_truth_label_id):
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"\nEvaluation Complete.")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

    results = {
        "run_config": run_config,
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_samples": total_predictions,
    }
    results_path = os.path.join(adapter_path, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✅ Evaluation results saved to {results_path}")

if __name__ == "__main__":
    main()