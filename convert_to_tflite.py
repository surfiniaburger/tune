# convert_to_tflite.py
import tensorflow as tf
from transformers import AutoModelForCausalLM
import torch
import os

# --- Configuration ---
# Path to the PyTorch model you just downloaded
pytorch_model_path = "./finetuned_model_for_conversion"
# Path where the intermediate TensorFlow model will be saved
tf_model_path = "./aura_mind_tf_model"

print("🚀 Starting PyTorch to TensorFlow Lite Conversion...")

# ==========================================================================
# --- Part 1: Convert PyTorch model to TensorFlow SavedModel ---
# ==========================================================================
print(f"\n[1/2] Converting PyTorch model at '{pytorch_model_path}' to TensorFlow...")

# Load the PyTorch model and immediately save it in the TensorFlow format.
# The 'from_pt=True' flag triggers the conversion.
# This is the step that failed on Kaggle but has the best chance here.
model_tf = AutoModelForCausalLM.from_pretrained(
    pytorch_model_path,
    from_pt=True,
    local_files_only=True, # Critical flag for local files
)
# Save the converted model in the full TensorFlow SavedModel format
model_tf.save_pretrained(tf_model_path, saved_model=True)
print(f"✅ Model successfully converted to TensorFlow SavedModel at '{tf_model_path}'")
del model_tf # Free up memory


# ==========================================================================
# --- Part 2: Convert TensorFlow SavedModel to TFLite with Quantization ---
# ==========================================================================
print(f"\n[2/2] Converting TensorFlow SavedModel to TFLite with INT8 quantization...")

# As per the documentation you found, we use TFLiteConverter.from_saved_model
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)

# And apply dynamic range quantization (INT8), which is the most recommended
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the final, quantized TFLite model
tflite_output_path = "agro_sage_maize_expert_int8.tflite"
with open(tflite_output_path, "wb") as f:
    f.write(tflite_quant_model)

print(f"\n🎉 SUCCESS! Quantized TFLite model saved to '{tflite_output_path}'")