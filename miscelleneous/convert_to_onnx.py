import torch
from pathlib import Path
import json
import os
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from optimum.exporters.onnx import export as onnx_export
from optimum.exporters.onnx.config import OnnxConfig
from optimum.utils import (
    DummyTextInputGenerator,
    DummyVisionInputGenerator,
    NormalizedTextConfig,
    NormalizedVisionConfig,
)
from typing import Dict

# --- Configuration ---
# Path to the PyTorch model you downloaded
pytorch_model_path = "/kaggle/input/model-name-auramind-maize-expert-e2b/pytorch/default/1/AuraMind-E2B-Finetuned-Sliced"
# Path where the final ONNX model will be saved
onnx_output_path = "./onnx_model_final"

print(f"PyTorch model path: {pytorch_model_path}")
print(f"ONNX output path: {onnx_output_path}")

# ==============================================================================
# CELL 1: DEFINE THE CUSTOM MULTIMODAL ONNX CONFIGURATION
# ==============================================================================
print("\nStep 1: Defining the custom multimodal ONNX configuration...")

class Gemma3nMultimodalOnnxConfig(OnnxConfig):
    """
    A custom ONNX configuration that explicitly handles both the vision and text
    modalities of the Gemma3n model.
    """
    def __init__(self, config: PretrainedConfig, task: str = "image-to-text", **kwargs):
        super().__init__(config, task=task, **kwargs)
        self.vision_config = NormalizedVisionConfig(config.vision_config)
        self.text_config = NormalizedTextConfig(config.text_config)

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs) -> Dict[str, torch.Tensor]:
        """Explicitly create dummy inputs for both vision and text.
        This is the key to handling complex multimodal models.
        """
        batch_size = kwargs.get("batch_size", 2)
        sequence_length = kwargs.get("sequence_length", 16)

        # Generate vision inputs
        vision_generator = DummyVisionInputGenerator(
            task=self.task,
            normalized_config=self.vision_config,
            batch_size=batch_size,
        )
        vision_inputs = vision_generator.generate()

        # Generate text inputs
        text_generator = DummyTextInputGenerator(
            task="text-generation", # Use a standard task for the text part
            normalized_config=self.text_config,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        text_inputs = text_generator.generate()

        # Combine inputs into the final dictionary expected by the model
        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": vision_inputs["pixel_values"],
        }

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "logits": {0: "batch_size", 1: "sequence_length"},
        }

# ==============================================================================
# CELL 2: PREPARE AND RUN THE ONNX EXPORT
# ==============================================================================
print("\nStep 2: Preparing and running the final ONNX export...")

try:
    # --- Step 2.1: Clean the Configuration ---
    print("   - Cleaning the model configuration...")
    config = AutoConfig.from_pretrained(pytorch_model_path, trust_remote_code=True)
    if hasattr(config, "quantization_config"): delattr(config, "quantization_config")
    if hasattr(config, "unsloth_fixed"): delattr(config, "unsloth_fixed")
    print("   - Configuration cleaned successfully.")

    # --- Step 2.2: Load the Model with the Cleaned Config ---
    print("   - Loading model with the cleaned configuration...")
    model = AutoModelForCausalLM.from_pretrained(
        pytorch_model_path,
        config=config,
        torch_dtype=torch.bfloat16, # Use bfloat16 to prevent memory overload
        trust_remote_code=True,
    )
    print("   - Model loaded successfully!")

    # --- Step 2.3: Configure and Run the ONNX Export ---
    print("   - Configuring the ONNX export for the full multimodal model...")
    onnx_config = Gemma3nMultimodalOnnxConfig(model.config)

    output_path_obj = Path(onnx_output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    print("   - Starting ONNX export...")
    onnx_export(
        model=model,
        config=onnx_config,
        output=output_path_obj / "model.onnx",
        opset=14,
    )
    print("\n✅ ONNX conversion process completed successfully!")
    print(f"   The exported model is saved in: {output_path_obj.resolve()}")

except Exception as e:
    print(f"\n❌ An error occurred during the ONNX conversion process: {e}")
    print("   Please check the following:")
    print("   1. Ensure all dependencies are installed correctly (`pip install \"optimum[exporters]\" transformers torch accelerate`).")
    print("   2. Verify that the `pytorch_model_path` is correct.")
