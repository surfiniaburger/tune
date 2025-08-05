import gradio as gr
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

import tempfile
import os

# Load model ONCE globally
model_path = "./finetuned_model_for_conversion"
model, processor = load(model_path)
config = model.config

def run_inference(image, audio):
    try:
        # Save audio and image temporarily
        image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(image_path.name)

        audio_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        with open(audio_path.name, "wb") as f:
            f.write(audio)

        # Prepare prompt
        prompt = "Describe what you see and hear in the provided files."
        formatted_prompt = apply_chat_template(
            processor, config, prompt,
            num_images=1,
            num_audios=1
        )

        # Generate output
        result = generate(
            model,
            processor,
            formatted_prompt,
            image=[image_path.name],
            audio=[audio_path.name],
            verbose=True
        )

        # Return text + metrics
        text = result.text
        metrics = {
            "Prompt tokens": result.prompt_tokens,
            "Generation tokens": result.generation_tokens,
            "Total tokens": result.total_tokens,
            "Prompt tokens/sec": round(result.prompt_tps, 2),
            "Generation tokens/sec": round(result.generation_tps, 2),
            "Peak memory (GB)": round(result.peak_memory, 2)
        }

        return text, metrics

    except Exception as e:
        return f"Error: {e}", {}

# Gradio UI
demo = gr.Interface(
    fn=run_inference,
    inputs=[
        gr.Image(type="pil", label="📸 Upload or take a photo of your maize"),
        gr.Audio( type="binary", label="🎤 Ask a question about the plant")
    ],
    outputs=[
        gr.Textbox(label="🧠 What the model understands"),
        gr.JSON(label="📊 Inference Metrics")
    ],
    title="🌽 Maize Health AI (Offline Demo)",
    description="Upload a maize image and ask your question using voice. The model will describe what it sees and hears.",
    live=True
)

if __name__ == "__main__":
    demo.launch()
