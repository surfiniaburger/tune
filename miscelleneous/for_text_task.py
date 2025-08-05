import torch
from pathlib import Path
import json
import os
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig
from optimum.exporters.onnx import export as onnx_export
from optimum.exporters.onnx.config import TextDecoderOnnxConfig
from optimum.utils import (
    NormalizedConfig,
    NormalizedTextConfig,
    DummyPastKeyValuesGenerator,
    DummyTextInputGenerator,
)
from typing import Dict

# --- Configuration ---
# Path to the PyTorch model you downloaded
pytorch_model_path = "/kaggle/input/model-name-auramind-maize-expert-e2b/pytorch/default/1/AuraMind-E2B-Finetuned-Sliced"
# Path where the final ONNX model will be saved
onnx_output_path = "./onnx_model"

print(f"PyTorch model path: {pytorch_model_path}")
print(f"ONNX output path: {onnx_output_path}")

# ==============================================================================
# CELL 1: DEFINE THE CUSTOM ONNX CONFIGURATION FOR TEXT-ONLY
# ==============================================================================
print("\nStep 1: Defining a custom ONNX configuration...")

class CustomGemma3NTextOnlyOnnxConfig(TextDecoderOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummyPastKeyValuesGenerator,
    )
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_layers="num_hidden_layers",
        num_attention_heads="num_attention_heads",
        hidden_size="hidden_size",
        vocab_size="vocab_size",
    )
    def __init__(self, config: PretrainedConfig, task: str = "default", **kwargs):
        # We only pass the text_config portion to the parent
        super().__init__(config=config.text_config, task=task, **kwargs)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        # Defines the inputs for the text-only model
        if self.use_past:
            return {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "past_sequence_length + 1"},
            }
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

# ==============================================================================
# CELL 2: PREPARE AND RUN THE ONNX EXPORT
# ==============================================================================
print("\nStep 2: Preparing and running the ONNX export...")

try:
    # --- Step 2.1: Clean the Configuration ---
    print("   - Cleaning the model configuration...")
    # Load the config using the standard method
    cleaned_config = AutoConfig.from_pretrained(pytorch_model_path, trust_remote_code=True)

    # Remove the Unsloth/quantization attributes that cause loading errors with float16 weights
    if hasattr(cleaned_config, "quantization_config"):
        delattr(cleaned_config, "quantization_config")
    if hasattr(cleaned_config, "unsloth_fixed"):
        delattr(cleaned_config, "unsloth_fixed")
    
    print("   - Configuration cleaned successfully.")

    # --- Step 2.2: Load the Model with the Cleaned Config ---
    print("   - Loading model with the cleaned configuration...")
    model = AutoModelForCausalLM.from_pretrained(
        pytorch_model_path,
        config=cleaned_config,
        torch_dtype=torch.float32, # Use float32 for better ONNX compatibility
        trust_remote_code=True,
    )
    print("   - Model loaded successfully!")

    # --- Step 2.3: Configure and Run the ONNX Export ---
    print("   - Configuring the ONNX export for the text model...")
    # We only care about the text part for this export
    custom_onnx_config = CustomGemma3NTextOnlyOnnxConfig(config=model.config, task="text-generation")

    print("   - Starting ONNX export...")
    # Use the lower-level `export` function which is better for pre-loaded models
    onnx_export(
        model=model,
        config=custom_onnx_config,
        output=Path(onnx_output_path),
        opset=14,
    )
    print("\n✅ ONNX conversion process completed successfully!")
    print(f"   The exported model is saved in: {Path(onnx_output_path).resolve()}")

except Exception as e:
    print(f"\n❌ An error occurred during the ONNX conversion process: {e}")
    print("   Please check the following:")
    print("   1. Ensure all dependencies are installed correctly (`pip install \"optimum[exporters]\" transformers torch accelerate`).")
    print("   2. Verify that the `pytorch_model_path` is correct.")
    print("   3. Your model might have a specific operator not supported by the default ONNX opset. You can try adjusting the `opset` parameter.")