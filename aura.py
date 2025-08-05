import streamlit as st
import os
import numpy

# Audio input
audio_file = st.audio_input("Record your audio message")

# Image input: upload only (no webcam)
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Save files if provided
audio_path = None
image_path = None

if audio_file:
    audio_path = "user_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_file.getbuffer())
    st.audio(audio_file)

if uploaded_image:
    image_path = "user_image.png"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    st.image(uploaded_image)

# Model inference
if st.button("Run Model") and audio_path and image_path:
    import mlx.core as mx
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model_path = "./finetuned_model_for_conversion"
    try:
        model, processor = load(model_path)
        config = model.config

        prompt = "Describe what you see and hear in the provided files."
        formatted_prompt = apply_chat_template(
            processor, config, prompt,
            num_images=1,
            num_audios=1
        )

        output = generate(
            model,
            processor,
            formatted_prompt,
            image=[image_path],
            audio=[audio_path],
            verbose=True
        )

        st.markdown("### Model Output")
        st.write(output)
    except FileNotFoundError:
        st.error(f"Error: Model not found at path '{model_path}'.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please record audio and provide an image to run the model.")