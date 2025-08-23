"""
==============================================================================
EVALUATE.PY
==============================================================================
This script evaluates a trained model adapter on the validation dataset.

It is called by `run_sweep.py` after a training run is complete.

Steps:
1. Loads the base model and the specific adapter from the latest training run.
2. Loads the pre-cached validation dataset.
3. Iterates through the validation set, generating predictions for each image.
4. Compares the model's prediction against the true label.
5. Calculates the overall accuracy.
6. Saves the evaluation results (accuracy) and the run's hyperparameters
   to a JSON file in the run's output directory.
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
BASE_MODEL_NAME = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"
CACHED_VAL_PATH = "data/cached_val_dataset_hf"
CONFIG_PATH = "current_run_config.json"

# --- 2. Helper function to extract the ground truth ---
def get_ground_truth(sample):
    """Extracts the ground truth label from the assistant's message."""
    # The ground truth is in the assistant's response from the original data creation
    # We need to re-create the text prompt to find it.
    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
    text_prompt = processor.tokenizer.apply_chat_template(
        sample["messages"], tokenize=False, add_generation_prompt=False
    )
    # Example assistant message: "...model\nThis is a Healthy Maize Plant."
    match = re.search(r"This is a (.*?)\\.", text_prompt)
    if match:
        return match.group(1).strip()
    return ""

# --- 3. Main Evaluation Function ---
def main():
    print("--- Starting Evaluation ---")

    # Load run configuration to find the correct adapter path
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Could not find {CONFIG_PATH}.")
    with open(CONFIG_PATH, 'r') as f:
        run_config = json.load(f)
    adapter_path = f"outputs/{run_config['run_name']}"
    print(f"Evaluating adapter: {adapter_path}")

    # Load model and processor
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

    # Load the cached validation dataset
    print(f"Loading cached validation set from {CACHED_VAL_PATH}")
    validation_dataset = load_from_disk(CACHED_VAL_PATH)
    # The pre-processed dataset doesn't have the original `messages` column,
    # so we need to load the raw data again to get ground truth.
    # This is a limitation of the current caching method.
    from cache_datasets import create_conversation_dict, LOCAL_VAL_PATH
    image_paths = list(LOCAL_VAL_PATH.glob("**/*.jpg")) + list(LOCAL_VAL_PATH.glob("**/*.jpeg"))
    raw_val_dataset = [create_conversation_dict(p, p.parent.name) for p in image_paths]
    raw_val_dataset = [item for item in raw_val_dataset if item is not None]

    correct_predictions = 0
    total_predictions = len(validation_dataset)

    # Iterate through the validation set
    for i, item in enumerate(tqdm(validation_dataset, desc="Evaluating")):
        # Get the raw image and ground truth from the parallel raw dataset
        raw_sample = raw_val_dataset[i]
        image = raw_sample["messages"][0]['content'][1]['image']
        ground_truth_label = get_ground_truth(raw_sample)

        # Prepare input using the robust, two-step method
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

        # Generate prediction
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Parse the final answer
        prompt_marker = "model\n"
        answer_start_index = response.rfind(prompt_marker)
        if answer_start_index != -1:
            final_answer = response[answer_start_index + len(prompt_marker):].strip()
            # Normalize the answer for comparison
            predicted_label = final_answer.replace("This is a ", "").replace(".", "").strip()
        else:
            predicted_label = ""

        # Compare prediction to ground truth
        if predicted_label == ground_truth_label:
            correct_predictions += 1

    # Calculate final accuracy
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"\nEvaluation Complete.")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Save results to a JSON file
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
