
# Aura-Mind: An AI Field Guide for the Nigerian Farmer

## The Spark: A Problem I Couldn't Ignore

This project didn't start in a sterile lab or a corporate boardroom. It started with a memory and a frustration. Growing up in Nigeria, I've always been surrounded by the vibrant, chaotic, and beautiful reality of our local agriculture. I've walked through markets like the one at Boundary, in Apapa, Lagos—a sprawling universe of produce where the hopes of countless farmers are laid out on display. But I've also seen the other side: the quiet desperation in a farmer's eyes when they see their maize leaves yellowing, their tomato plants wilting, their peppers succumbing to a blight they can't name.

This isn't a small problem. Agriculture is the backbone of our economy, employing over [**70% of Nigeria's population**](https://www.google.com/search?q=percentage+of+nigerian+population+in+agriculture), yet it's a sector fighting an uphill battle. We face an estimated [**30-40% in crop losses**](https://www.google.com/search?q=crop+losses+in+nigeria+due+to+pests+and+diseases) annually due to pests and diseases. For a smallholder farmer, that's not just a statistic; it's the difference between sending their children to school and another year of struggle. The knowledge to fight back exists, but it's locked away in academic papers and expert manuals, inaccessible to those who need it most, especially in rural areas where the biggest barrier isn't just the cost of data, but the near-total lack of reliable internet connectivity.

The spark for this project came from a paper by Professor T. O. Anjorin from the University of Abuja, which demonstrated the potential of using deep learning to identify local crop diseases. It was a brilliant proof of concept, but it highlighted a gap: how do we take this powerful lab-based technology and put it directly into the hands of a farmer in a remote village, in a way that works offline, speaks their language, and understands their reality?

**This became my mission: to build a tool that was personal, private, and powerful enough to run in the palm of a farmer's hand.**

## The Proposed Solution: An AI That Lives on the Edge

I envisioned Aura-Mind: an offline-first, AI-powered companion that could run on a basic Android phone. A farmer could simply take a picture of a struggling plant, and the AI would provide a probable diagnosis and simple, actionable, low-cost advice—no internet required.

The launch of Google's Gemma 3N, a powerful multimodal model designed for on-device operation, was the final piece of the puzzle. The technology was finally here. The challenge was to bend it to our will.

## The Journey: A Trial by Fire

The path from idea to a working model has been a brutal, beautiful, and deeply personal fight. It has been a microcosm of the very problem we're trying to solve: a battle against constraints, a search for knowledge, and an absolute refusal to quit.

#### The First Hurdle: Data, The Real Gold

The academic dataset from Prof. Anjorin's paper was a fantastic starting point, but I quickly realized it wasn't enough. It was clean, academic. I needed the messiness of the real world. I went to the market at Boundary, Apapa, with my phone. I didn't just take pictures of perfect, diseased leaves; I photographed the rejected piles. The bruised tomatoes, the spotted peppers, the maize cobs with their husks half-torn, sitting in wicker baskets under the harsh Lagos sun. I spoke to the sellers, the real experts, and gathered not just images, but context. This project is built on that ground truth.

#### The Second Hurdle: The Tools and the "Goliath" of Complexity

My initial plan was to use Apple's MLX framework on my MacBook. It felt like the perfect "David vs. Goliath" story—building a powerful AI on a personal computer. But the libraries were too new, the dependencies too fragile. It was a dead end.

I pivoted to the cloud, using free tiers on Google Colab and Kaggle. This is where the real war began. The next few weeks were a relentless cycle of debugging:
*   We fought **dependency hell**, with `protobuf` and `tensorflow` versions clashing in the dead of night.
*   We battled **cryptic hardware errors**, like the infamous `CUDA:1 and cpu` error on Kaggle's multi-GPU machines, a problem that forced us to understand the deep, internal workings of device mapping.
*   We were mocked by the **"Silent Hang,"** where a training job would run for hours, GPUs blazing at 0%, because of a subtle I/O bottleneck from loading high-resolution images. We solved it by building a robust pre-processing pipeline, only to discover a better way.

#### The Breakthrough: A Community and a Clue

The turning point came from the Unsloth community. After weeks of fighting a broken evaluation pipeline—where our model showed a perfect training loss but a 0% accuracy—I found a Discord discussion. The developers themselves confirmed it: a known architectural quirk in the fine-tuned Gemma 3N meant our evaluation method would never work.

This was a painful, liberating discovery. Our model wasn't broken; our measuring stick was. We pivoted again, armed with the developer's own examples, and rebuilt the training pipeline the "Unsloth Way," using their custom `UnslothVisionDataCollator`.

The result was a training run that completed **18 epochs in under an hour** on a free Kaggle GPU. The loss curve was a thing of beauty, plummeting from ~13.0 to absolute zero. We had done it. We had a trained, expert model.

## Where We Are Now, and the Fire That Remains

Today, I have a powerful, fine-tuned "Maize Expert" model, safely stored on the Hugging Face Hub. I have a proven, robust training pipeline. But the journey is not over.

The final hurdle, converting this PyTorch model into the `.task` format for Android, has revealed the final frontier. The official conversion tools, from `ai-edge-torch` to `optimum`, are not yet mature enough to handle this specific, brand-new, fine-tuned multimodal architecture. They fail with deep, unrecoverable errors.

It hurts. To be so close to the finish line and be blocked by the very tools meant to get you there is a special kind of frustration.

But it is not a defeat. It is a diagnosis.

This project is no longer just about building an app. It is about stress-testing the entire on-device ecosystem and finding its breaking points. It's about contributing back to the community by documenting this incredibly difficult journey so the next developer doesn't have to fight the same battles.

My passion to see this through has only intensified. The next steps are clear:
1.  **Showcase the Power:** I will build an interactive Gradio demo to prove the incredible capability of the trained model. This will be the centerpiece of my submission.
2.  **Engage the Experts:** I will go back to Professor Anjorin, not with an idea, but with a working AI and a story of our struggle. I will re-engage the developers of the conversion tools, not with a bug report, but with a full, reproducible case study.
3.  **Find a Way:** (explore alternatives like gguf) I will not stop until I find a way to get this model onto a phone. Whether it's waiting for the tools to mature, or diving into the C++ source code of the TensorFlow Lite converter myself, I will find a path.

The problem is too important. The need is too great. And we are too close to give up now. Let's go.





# convert_with_aiedge.py
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from pathlib import Path
import ai_edge_torch
# As per the official documentation, use `quant_recipes` for quantization.
from ai_edge_torch.generative.quantize import quant_recipes
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

    # --- 2. Create Dummy Inputs for Tracing ---
    print("\n[2/4] Creating dummy inputs for tracing...")
    # We need to create a sample input that matches the model's forward signature.
    # This logic is adapted from your previous ONNX export script.
    batch_size = 1
    text_seq_len = 16  # A standard length for a dummy prompt

    # Get model-specific dimensions from the config
    vocab_size = config.text_config.vocab_size
    # This attribute lives on the main config, not the vision_config sub-object.
    # It's a common pattern for parameters that bridge the modalities.
    num_image_tokens = config.vision_soft_tokens_per_image
    image_token_id = config.image_token_id
    # The `AttributeError` indicates `num_channels` is not in the vision_config.
    # This can happen if it wasn't saved during fine-tuning. For standard
    # image models, this value is almost always 3 (for RGB), so we can safely hardcode it.
    num_channels = 3
    # The `AttributeError` indicates `image_size` is also not in the vision_config.
    # Based on previous scripts and common model architectures, we can safely hardcode this to 768.
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

    # d) Assemble the sample inputs tuple in the correct order for the model's forward pass
    # The signature is forward(self, input_ids, pixel_values, attention_mask, ...)
    # We pass `None` for the optional `input_features` (audio) argument.
    sample_inputs = (final_input_ids, pixel_values, None, final_attention_mask)
    print("✅ Dummy inputs created.")

    # --- 3. Convert, Quantize, and Export to TFLite ---
    print("\n[3/4] Converting, quantizing, and exporting to TFLite...")

    # The TypeError confirms the previous QuantConfig instantiation was incorrect.
    # The official, documented way is to use a pre-defined recipe.
    # `full_int8_dynamic_recipe` provides the settings for dynamic INT8 quantization.
    quant_config = quant_recipes.full_int8_dynamic_recipe()

    # The `convert` function handles tracing, quantization, and returns an Edge model.
    edge_model = ai_edge_torch.convert(model, sample_inputs, quant_config=quant_config)

    edge_model.export(TFLITE_OUTPUT_PATH)
    print(f"\n🎉 SUCCESS! Quantized TFLite model saved to: {TFLITE_OUTPUT_PATH}")
    print(f"   File size: {TFLITE_OUTPUT_PATH.stat().st_size / 1e6:.2f} MB")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")
    traceback.print_exc()