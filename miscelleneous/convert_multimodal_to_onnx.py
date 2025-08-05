import torch
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from optimum.exporters.onnx import export as onnx_export
from optimum.exporters.onnx.config import TextDecoderOnnxConfig
from optimum.utils import (
    DummyVisionInputGenerator,
    DummyTextInputGenerator,
    DummyPastKeyValuesGenerator,
    NormalizedTextConfig,
    NormalizedVisionConfig,
)
from typing import Dict
import gc

# --- Configuration ---
# Path to the SLICED model from your second notebook. This is the only one that will fit in memory.
pytorch_model_path = "/kaggle/working/AuraMind-E2B-Finetuned-Sliced/"

# Path where the final ONNX model will be saved
onnx_output_path = "./onnx_multimodal_model"

print(f"Targeting SLICED model path: {pytorch_model_path}")
print(f"ONNX output path: {onnx_output_path}")

# ==============================================================================
# CELL 1: DEFINE THE CUSTOM ONNX CONFIGURATION FOR IMAGE-TO-TEXT
# ==============================================================================
print("\nStep 1: Defining a custom ONNX configuration for the image-to-text model...")

class CustomGemma3NImageToTextOnnxConfig(TextDecoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_layers="num_hidden_layers",
        num_attention_heads="num_attention_heads",
        hidden_size="hidden_size",
    )

    def __init__(self, config: PretrainedConfig, task: str = "default", **kwargs):
        super().__init__(config=config.text_config, task=task, **kwargs)
        self.text_config = config.text_config
        self.vision_config = config.vision_config
        self.config = config

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        text_inputs = super().inputs
        vision_inputs = {
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}
        }
        return {**vision_inputs, **text_inputs}

    # Using a minimal sequence length to conserve memory during export
    def generate_dummy_inputs(self, batch_size: int = 1, sequence_length: int = 260, **kwargs) -> Dict[str, torch.Tensor]:
        text_input_generator = DummyTextInputGenerator(
            self.task, self._normalized_config, batch_size=batch_size, sequence_length=sequence_length, **self.text_config.to_dict(),
        )
        dummy_inputs = {
            "input_ids": text_input_generator.generate(input_name="input_ids", framework="pt"),
            "attention_mask": text_input_generator.generate(input_name="attention_mask", framework="pt"),
        }
        image_token_id = self.config.image_token_id
        tokens_per_image = self.config.vision_soft_tokens_per_image
        if sequence_length < tokens_per_image:
            raise ValueError(f"Sequence length must be at least {tokens_per_image} to hold image tokens.")
        for i in range(batch_size):
            dummy_inputs["input_ids"][i, :tokens_per_image] = image_token_id
        if self.use_past:
            past_key_values_generator = DummyPastKeyValuesGenerator(
                self.task, self._normalized_config, batch_size=batch_size, sequence_length=sequence_length,
            )
            dummy_inputs.update(past_key_values_generator.generate(framework="pt"))
        normalized_vision_config = NormalizedVisionConfig(self.vision_config)
        vision_input_generator = DummyVisionInputGenerator(
            self.task, normalized_vision_config, batch_size=batch_size, num_channels=3, height=224, width=224,
        )
        dummy_inputs["pixel_values"] = vision_input_generator.generate(input_name="pixel_values", framework="pt")
        return dummy_inputs

print("   - CustomGemma3NImageToTextOnnxConfig defined.")

# ==============================================================================
# CELL 2: PREPARE AND RUN THE ONNX EXPORT
# ==============================================================================
print("\nStep 2: Preparing and running the ONNX export...")

try:
    # --- Step 2.1: Load, Clean, and Fix the Sliced Model's Configuration ---
    print("   - Loading and fixing configuration for the SLICED model...")
    config = AutoConfig.from_pretrained(pytorch_model_path, trust_remote_code=True)
    
    # Clean the config by removing Unsloth and quantization artifacts
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
        print("   - Removed 'quantization_config'.")
    if hasattr(config, "unsloth_fixed"):
        delattr(config, "unsloth_fixed")
        print("   - Removed 'unsloth_fixed'.")

    # Manually set the model_type to fix the recognition issue
    config.model_type = "gemma3n"
    print(f"   - Manually set model_type to: {config.model_type}")

    # --- Step 2.2: Load the Sliced Model with the Corrected Config ---
    print("   - Loading SLICED model with the corrected configuration...")
    # Using float16 to reduce memory footprint
    model = AutoModelForCausalLM.from_pretrained(
        pytorch_model_path,
        config=config, # Pass the fixed config object here
        torch_dtype=torch.float16, 
        trust_remote_code=True,
    )
    model.eval()
    print("   - Sliced model loaded successfully!")

    # --- Step 2.3: Configure and Run the ONNX Export ---
    print("   - Configuring the ONNX export...")
    custom_onnx_config = CustomGemma3NImageToTextOnnxConfig(
        config=model.config, 
        task="text-generation"
    )

    # --- Step 2.4: Clean up memory before export ---
    print("   - Cleaning up memory before starting export...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("   - Starting ONNX export...")
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
    import traceback
    traceback.print_exc()
