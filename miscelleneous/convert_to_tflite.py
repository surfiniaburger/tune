# convert_to_tflite.py
import tensorflow as tf
from transformers import AutoConfig
from optimum.exporters.tflite import export, TFLiteConfig, validate_model_outputs
from optimum.exporters.tasks import TasksManager
from optimum.utils import NormalizedConfigManager
from pathlib import Path
import os, traceback

# --- Configuration ---
PYTORCH_MODEL_PATH = "./finetuned_model_for_conversion"
TFLITE_OUTPUT_PATH = Path("./aura_mind_maize_expert.tflite")

print("🚀 Starting PyTorch to TensorFlow Lite Conversion...")
print(f"   - PyTorch Model: {PYTORCH_MODEL_PATH}")
print(f"   - TFLite Output: {TFLITE_OUTPUT_PATH}")

# ==============================================================================
# 1. DEFINE A CUSTOM TFLITE CONFIGURATION FOR GEMMA-3N
# ==============================================================================
print("\n[1/3] Defining custom TFLite configuration...")

class Gemma3nTFLiteConfig(TFLiteConfig):
    """
    A custom TFLiteConfig for the Gemma-3n model. This tells the exporter
    what inputs and outputs to expect.
    """
    # This maps the model's config values to a standardized format.
    # We only need `hidden_size` for this model.
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("gemma")

    # This defines the inputs the TFLite model will accept.
    # We use `None` for dynamic dimensions (batch_size, sequence_length).
    @property
    def inputs_specs(self) -> list[tf.TensorSpec]:
        return [
            tf.TensorSpec((None, None), tf.int64, name="input_ids"),
            tf.TensorSpec((None, None), tf.int64, name="attention_mask"),
            # The pixel_values input is for the image.
            tf.TensorSpec((None, 3, 768, 768), tf.float32, name="pixel_values"),
        ]

    # This defines the output(s) of the TFLite model.
    @property
    def outputs(self) -> list[str]:
        return ["logits"]


# ==============================================================================
# 2. LOAD THE PYTORCH MODEL AS A TENSORFLOW MODEL
# ==============================================================================
print("\n[2/3] Loading PyTorch model into TensorFlow...")

# The `from_pt=True` flag is the magic that converts the model.
# This requires both PyTorch and TensorFlow to be installed.
try:
    # First, load the config to get the model type
    config = AutoConfig.from_pretrained(PYTORCH_MODEL_PATH, trust_remote_code=True)

    # The generic `TFAutoModelForVision2Seq` (which is what `TasksManager` returns for
    # "image-to-text") does not know about the custom `Gemma3nConfig`.
    # We must bypass the auto-class lookup and use the specific TF model class directly.
    try:
        from transformers import TFGemma3nForConditionalGeneration as model_class
    except ImportError:
        print("❌ Could not import TFGemma3nForConditionalGeneration. Make sure your `transformers` version supports it.")
        exit()

    # Now load the model, triggering the conversion from PyTorch
    model_tf = model_class.from_pretrained(PYTORCH_MODEL_PATH, from_pt=True, trust_remote_code=True)
    print("✅ Model successfully loaded into TensorFlow.")

except Exception as e:
    print(f"❌ Failed to load PyTorch model into TensorFlow: {e}")
    print("   Please ensure both `torch` and `tensorflow` are installed.")
    exit()

# ==============================================================================
# 3. EXPORT TO TFLITE WITH QUANTIZATION
# ==============================================================================
print("\n[3/3] Exporting to TFLite with INT8 dynamic range quantization...")

try:
    # Instantiate our custom config
    tflite_config = Gemma3nTFLiteConfig(model_tf.config)

    # Export the model
    export(
        model=model_tf,
        config=tflite_config,
        output=TFLITE_OUTPUT_PATH,
        quantize="int8-dynamic" # Apply INT8 dynamic range quantization
    )

    print(f"\n🎉 SUCCESS! Quantized TFLite model saved to '{TFLITE_OUTPUT_PATH}'")
    print(f"   File size: {TFLITE_OUTPUT_PATH.stat().st_size / 1e6:.2f} MB")

except Exception as e:
    print(f"❌ TFLite export failed: {e}")
    traceback.print_exc()