"""
==============================================================================
EVALUATE.PY (v2 - Robust Scoring)
==============================================================================
This script evaluates a trained model adapter on the validation dataset.
It now includes a robust, keyword-based scoring mechanism inspired by the
Weave implementation to prevent failures due to minor phrasing differences.

It is called by `run_sweep.py` after a training run is complete.
"""

import os
import json
import torch
from datasets import load_from_disk
from unsloth import FastVisionModel
from transformers import AutoProcessor
from tqdm import tqdm
import re

# --- 1. Configuration ---
BASE_MODEL_NAME = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"
CACHED_VAL_PATH = "data/cached_val_dataset_hf"
CONFIG_PATH = "current_run_config.json"

# --- 2. Helper Functions ---
def get_ground_truth(sample):
    """Extracts the ground truth label from the assistant's message."""
    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
    text_prompt = processor.tokenizer.apply_chat_template(
        sample["messages"], tokenize=False, add_generation_prompt=False
    )
    match = re.search(r"This is a (.*?)\\.", text_prompt)
    if match:
        return match.group(1).strip()
    return ""

def is_prediction_correct(prediction: str, ground_truth: str) -> bool:
    """
    Calculates accuracy by checking for keywords in the model's prediction,
    making it robust to phrasing changes.
    """
    pred_lower = prediction.lower()
    gt_lower = ground_truth.lower()

    target_keyword = ""
    if "healthy" in gt_lower:
        target_keyword = "healthy"
    elif "phosphorus" in gt_lower:
        target_keyword = "phosphorus"

    if not target_keyword:
        return False # Cannot determine ground truth

    # A prediction is correct if it contains BOTH "maize" and the target keyword
    return "maize" in pred_lower and target_keyword in pred_lower

# --- 3. Main Evaluation Function ---
def main():
    print("-- Starting Evaluation --")

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Could not find {CONFIG_PATH}.")
    with open(CONFIG_PATH, 'r') as f:
        run_config = json.load(f)
    adapter_path = f"outputs/{run_config['run_name']}"
    print(f"Evaluating adapter: {adapter_path}")

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
    from cache_datasets import create_conversation_dict, LOCAL_PATHS
    LOCAL_VAL_PATH = LOCAL_PATHS["validation"]
    image_paths = list(LOCAL_VAL_PATH.glob("**/*.jpg")) + list(LOCAL_VAL_PATH.glob("**/*.jpeg"))
    raw_val_dataset = [create_conversation_dict(p, p.parent.name) for p in image_paths]
    raw_val_dataset = [item for item in raw_val_dataset if item is not None]

    correct_predictions = 0
    total_predictions = len(validation_dataset)

    for i, item in enumerate(tqdm(validation_dataset, desc="Evaluating")):
        raw_sample = raw_val_dataset[i]
        image = raw_sample["messages"][0]['content'][1]['image']
        ground_truth_label = get_ground_truth(raw_sample)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify the condition of this maize plant. Choose from: Healthy Maize Plant, Maize Phosphorus Deficiency."},
                    {"type": "image", "image": image},
                ],
            }
        ]
        text_prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=text_prompt, images=image, return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=20, use_cache=True)
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        prompt_marker = "model\n"
        answer_start_index = response.rfind(prompt_marker)
        if answer_start_index != -1:
            final_answer = response[answer_start_index + len(prompt_marker):].strip()
        else:
            final_answer = ""

        # Use the new robust scoring function
        if is_prediction_correct(final_answer, ground_truth_label):
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
    print(f"Evaluation results saved to {results_path}")

if __name__ == "__main__":
    main()
