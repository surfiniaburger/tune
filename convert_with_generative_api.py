# convert_with_generative_api.py
import torch
from absl import app
from absl import flags
from transformers import AutoModelForCausalLM

# Import the necessary, specific tools
from ai_edge_torch.generative.utilities import converter as generative_converter
from ai_edge_torch.generative.utilities.export_config import Gemma3Config

# --- Define Command-Line Arguments ---
generative_converter.define_conversion_flags('aura_mind_maize')
FLAGS = flags.FLAGS

def main(_):
  if not FLAGS.checkpoint_path or not FLAGS.output_path:
    raise ValueError('Must specify --checkpoint_path and --output_path')

  # --- Load our complete, fine-tuned multimodal model ---
  print(f"Loading full multimodal model from: {FLAGS.checkpoint_path}")
  pytorch_model = AutoModelForCausalLM.from_pretrained(
        FLAGS.checkpoint_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
  print("✅ Full model loaded.")

  # ============================================================================
  # --- Use the High-Level Generative Converter with EXPLICIT Arguments ---
  # ============================================================================
  print("\n🔥 Starting TFLite conversion for the full multimodal model...")

  # 1. Define the image input size. The vision encoder in Gemma 3N produces a
  #    fixed-size embedding. We can get this from the model's config.
  try:
      pixel_values_embedding_size = pytorch_model.vision_tower.config.hidden_size
  except AttributeError:
      print("Warning: Could not determine embedding size from config. Defaulting to 2048.")
      pixel_values_embedding_size = 2048

  pixel_values_size_for_converter = (1, pixel_values_embedding_size)
  print(f"Determined pixel_values_size for converter: {pixel_values_size_for_converter}")

  pixel_seq_len_for_converter = 256

  # 2. Create the ExportConfig object
  export_config = Gemma3Config(
      mask_as_input=FLAGS.mask_as_input,
      transpose_kv_cache=FLAGS.transpose_kv_cache,
  )

  # 3. Call the function with every required argument.
  generative_converter.convert_to_tflite(
        pytorch_model=pytorch_model, output_path=FLAGS.output_path, output_name_prefix=FLAGS.output_name_prefix,
        prefill_seq_len=FLAGS.prefill_seq_lens, kv_cache_max_len=FLAGS.kv_cache_max_len, quantize=FLAGS.quantize,
        pixel_values_size=pixel_values_size_for_converter, pixel_seq_len=pixel_seq_len_for_converter,
        lora_ranks=FLAGS.lora_ranks, export_config=export_config,
    )
  print("✅ Full multimodal model TFLite conversion complete.")
  print(f"Check your output path '{FLAGS.output_path}' for the .tflite file(s).")

if __name__ == "__main__":
    app.run(main)
