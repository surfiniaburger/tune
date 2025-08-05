# convert_with_aiedge.py
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from pathlib import Path
import ai_edge_torch
# As per the official documentation, use `quant_recipes` for quantization.
from ai_edge_torch.generative.quantize import quant_recipes
import gc
import traceback

# --- Configuration ---
PYTORCH_MODEL_PATH = "./finetuned_model_for_conversion"
TFLITE_OUTPUT_PATH = Path("./aura_mind_maize_expert_aiedge.tflite")

print("🚀 Starting PyTorch to TFLite Conversion with AI Edge Torch...")
print(f"   - PyTorch Model: {PYTORCH_MODEL_PATH}")
print(f"   - TFLite Output: {TFLITE_OUTPUT_PATH}")

try:
    # --- 1. Load PyTorch Model ---
    print("\n[1/4] Loading PyTorch model...")
    config = AutoConfig.from_pretrained(PYTORCH_MODEL_PATH, trust_remote_code=True)
    # Load the model in float32, as this is standard for conversion.
    model = AutoModelForCausalLM.from_pretrained(
        PYTORCH_MODEL_PATH,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use float16 to reduce memory usage by half
        low_cpu_mem_usage=True,     # Use less memory during loading
    ).eval()
    print("✅ Model loaded.")

    # --- Force garbage collection before tracing ---
    print("\n[2/4] Forcing garbage collection to free up memory...")
    gc.collect()
    print("✅ Garbage collection complete.")

    # --- 3. Create Dummy Inputs and Convert under inference_mode ---
    print("\n[3/4] Creating dummy inputs and converting model...")
    # The `zsh: killed` error is an Out-Of-Memory error from the OS.
    # Using torch.inference_mode() is crucial for memory-intensive exports.
    # It disables all gradient calculations, significantly reducing the memory
    # footprint of the tracing process inside ai_edge_torch.convert().
    with torch.inference_mode():
        # We need to create a sample input that matches the model's forward signature.
        batch_size = 1
        text_seq_len = 16  # A standard length for a dummy prompt

        # Get model-specific dimensions from the config
        vocab_size = config.text_config.vocab_size
        num_image_tokens = config.vision_soft_tokens_per_image
        image_token_id = config.image_token_id
        num_channels = 3
        image_size = 768
        height = image_size
        width = image_size

        # a) Create dummy text inputs
        input_ids = torch.randint(low=3, high=vocab_size, size=(batch_size, text_seq_len), dtype=torch.long)
        input_ids[:, 0] = config.bos_token_id if hasattr(config, "bos_token_id") else 2

        # b) Inject the special image token placeholder into the text
        part1 = input_ids[:, :1]
        part2 = input_ids[:, 1 + num_image_tokens:]
        image_token_tensor = torch.full((batch_size, num_image_tokens), image_token_id, dtype=torch.long)
        final_input_ids = torch.cat([part1, image_token_tensor, part2], dim=1)
        final_attention_mask = torch.ones_like(final_input_ids)

        # c) Create dummy vision inputs
        pixel_values = torch.randn(batch_size, num_channels, height, width, dtype=torch.float32)

        # d) Assemble the sample inputs tuple
        sample_inputs = (final_input_ids, pixel_values, None, final_attention_mask)
        print("✅ Dummy inputs created.")

        # --- 4. Quantize and Export to TFLite ---
        print("\n[4/4] Quantizing and exporting to TFLite...")
        quant_config = quant_recipes.full_int8_dynamic_recipe()
        edge_model = ai_edge_torch.convert(model, sample_inputs, quant_config=quant_config)
        edge_model.export(TFLITE_OUTPUT_PATH)

    print(f"\n🎉 SUCCESS! Quantized TFLite model saved to: {TFLITE_OUTPUT_PATH}")
    print(f"   File size: {TFLITE_OUTPUT_PATH.stat().st_size / 1e6:.2f} MB")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")
    traceback.print_exc()