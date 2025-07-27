# convert_with_mediapipe.py
import os

# This environment variable disables a plugin-loading mechanism in PyTorch that can cause issues on some systems.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import mediapipe as mp
from mediapipe.tasks.python.genai import converter
from pathlib import Path

# --- Configuration ---

# Path to the PyTorch model you downloaded from Hugging Face
CHECKPOINT_PATH = "./finetuned_model_for_conversion"

# The directory where the final .task file will be saved
OUTPUT_DIR = Path("./mediapipe_model")

# The base name for the output file (e.g., "aura_mind_maize_expert.task")
OUTPUT_NAME = "aura_mind_maize_expert"

def main():
    """
    Converts a Hugging Face checkpoint to a MediaPipe .task file.
    """
    print("🚀 Starting model conversion with the MediaPipe GenAI Converter...")
    print(f"   - Checkpoint Path: {CHECKPOINT_PATH}")
    print(f"   - Output Directory: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Create the ConversionConfig
    # This config tells the converter where to find the model and how to package it.
    # We use the specific arguments required by the MediaPipe converter.
    config = converter.ConversionConfig(
        input_ckpt=CHECKPOINT_PATH,
        ckpt_format="safetensors",
        model_type="GEMMA_3N_E2B_IT",
        output_dir=str(OUTPUT_DIR),
        # The output filename must be specified with `output_tflite_file`.
        # The `.task` extension is handled by the converter.
        output_tflite_file=str(OUTPUT_DIR / f"{OUTPUT_NAME}.task"),
        # Point to the directory containing tokenizer.json for bundling.
        vocab_model_file=CHECKPOINT_PATH,
        backend="cpu",  # Use 'gpu' if you have a GPU-specific model and target
    )

    # 2. Run the conversion
    try:
        converter.convert_checkpoint(config)
        print(f"\n✅ Successfully created '{OUTPUT_DIR / (OUTPUT_NAME + '.task')}'!")
        print("This file is ready to be used with the MediaPipe LLM Inference API on Android.")
    except Exception as e:
        print(f"\n❌ An error occurred during MediaPipe conversion: {e}")

if __name__ == "__main__":
    main()