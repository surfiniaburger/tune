import tensorflow as tf
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.onnx.config import OnnxConfig
from transformers.onnx.convert import export
from transformers.utils.generic import TensorType
from collections import OrderedDict
from pathlib import Path
import torch
import onnx
import tf2onnx
import os
from typing import Dict, Any, Optional

# --- Configuration ---
# Path to the PyTorch model you just downloaded
pytorch_model_path = "./finetuned_model_for_conversion"
# Path where the intermediate ONNX model will be saved
onnx_model_path = "./aura_mind.onnx"
# Path where the intermediate TensorFlow model will be saved
tf_model_path = "./aura_mind_tf_model"
# Path for the final TFLite model
tflite_output_path = "agro_sage_maize_expert_int8.tflite"

print("🚀 Starting PyTorch to TensorFlow Lite Conversion via transformers.onnx...")

# ==========================================================================
# --- Part 1: Convert PyTorch model to ONNX ---
# ==========================================================================
print(f"\n[1/3] Exporting PyTorch model at '{pytorch_model_path}' to ONNX...")

# For multimodal models, we need a processor that handles both text and images.
processor = AutoProcessor.from_pretrained(pytorch_model_path, local_files_only=True, trust_remote_code=True
)
model_pt = AutoModelForCausalLM.from_pretrained(
    pytorch_model_path,
    local_files_only=True,
    trust_remote_code=True,
    # Force the model to use a simpler, traceable attention mechanism.
    attn_implementation="eager",
)

# --- FIX ---
# The default causal mask creation uses torch.vmap, which can fail during
# ONNX tracing. Setting this flag forces a simpler, legacy implementation
# that is more compatible with the ONNX exporter.
model_pt.config.text_config._use_legacy_causal_mask = True
# --- END FIX ---

model_pt.eval()  # Set model to evaluation mode

# Define a custom ONNX configuration for our multimodal model.
# This tells the `transformers.onnx` exporter how to handle the model's
# inputs and outputs, replacing the need for manual dummy inputs.
class Gemma3nOnnxConfig(OnnxConfig):
    def generate_dummy_inputs(
        self,
        preprocessor,
        batch_size: int = -1,
        seq_length: int = -1,
        num_choices: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
        num_channels: int = 3,
        image_width: int = 40,
        image_height: int = 40,
        sampling_rate: int = 22050,
        time_duration: float = 5.0,
        frequency: int = 220,
        tokenizer=None,
    ) -> "OrderedDict[str, Any]":
        """
        Generates a valid set of dummy inputs for the Gemma3n model.
        This is necessary because the model has a complex, multimodal input structure.
        """
        if framework is None:
            framework = TensorType.PYTORCH

        dummy_image = torch.randn(1, 3, 768, 768, dtype=torch.float32)
        # For multimodal models, it's more robust to use a chat template.
        # This ensures special tokens like `<image>` are handled correctly.
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the condition of this maize plant?"},
                    {"type": "image"},
                ],
            },
        ]
        prompt = preprocessor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        # The processor is callable, but the type hints are loose, so we ignore the lint warning.
        inputs = preprocessor(  # type: ignore
            text=prompt, images=dummy_image, return_tensors=framework.lower()
        )
        # The ONNX exporter's JIT tracer expects a plain dict to treat it as kwargs.
        return dict(inputs)

    @property
    def inputs(self) -> "OrderedDict[str, Dict[int, str]]":
        # This defines the inputs the ONNX model will expect.
        return OrderedDict(
            [
                ("input_ids", {0: "batch_size", 1: "sequence_length"}),
                ("attention_mask", {0: "batch_size", 1: "sequence_length"}),
                ("pixel_values", {0: "batch_size", 2: "height", 3: "width"}),
            ]
        )

    @property
    def outputs(self) -> "OrderedDict[str, Dict[int, str]]":
        # This defines the outputs the ONNX model will produce.
        return OrderedDict(
            [
                ("logits", {0: "batch_size", 1: "sequence_length"}),
            ]
        )

# Instantiate the config
onnx_config = Gemma3nOnnxConfig(model_pt.config)

# Use the transformers.onnx.export function
# This function is specifically designed to handle the complexities of
# transformers models and is more robust than the standard torch.onnx.export.
export(
    preprocessor=processor,
    model=model_pt,
    config=onnx_config,
    opset=15, # Use a modern opset
    output=Path(onnx_model_path),
)

print(f"✅ Model successfully exported to ONNX at '{onnx_model_path}'")
del model_pt  # Free up memory

# ==========================================================================
# --- Part 2: Convert ONNX to TensorFlow SavedModel ---
# ==========================================================================
print(f"\n[2/3] Converting ONNX model to TensorFlow SavedModel...")

# Load the ONNX model
onnx_model = onnx.load(onnx_model_path)

# Convert ONNX to TensorFlow
# Note: tf2onnx might need specific opset versions. If 17 fails, try 13 or 15.
tf_rep = tf2onnx.prepare.prepare_onnx_graph(onnx_model, opset=17)
tf_rep.export_graph(tf_model_path)

print(f"✅ Model successfully converted to TensorFlow SavedModel at '{tf_model_path}'")
del onnx_model  # Free up memory

# ==========================================================================
# --- Part 3: Convert TensorFlow SavedModel to TFLite with Quantization ---
# ==========================================================================
print(f"\n[3/3] Converting TensorFlow SavedModel to TFLite with INT8 quantization...")

converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)

# And apply dynamic range quantization (INT8), which is the most recommended
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the final, quantized TFLite model
with open(tflite_output_path, "wb") as f:
    f.write(tflite_quant_model)

print(f"\n🎉 SUCCESS! Quantized TFLite model saved to '{tflite_output_path}'")