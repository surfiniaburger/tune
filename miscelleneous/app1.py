import streamlit as st

st.set_page_config(layout="wide")

st.title("Aura-Mind: An Offline-First AI Companion for Farmers")

st.markdown("""
### The Spark: A Problem I Couldn't Ignore
""")


# Add a title for the inference section
st.title("Aura-Mind Inference")

# Add a file uploader for the image
image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Add a file uploader for the audio
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# Add a button to run the inference
if st.button("Run Inference"):
    if image_file is not None and audio_file is not None:
        # Save the uploaded files to a temporary location
        with open("temp_image.png", "wb") as f:
            f.write(image_file.getbuffer())
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file.getbuffer())

        # Run the inference
        import mlx.core as mx
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template

        model_path = "./finetuned_model_for_conversion"

        try:
            model, processor = load(model_path)
            config = model.config

            image_path = ["temp_image.png"]
            audio_path = ["temp_audio.wav"]

            prompt = "Describe what you see and hear in the provided files."

            formatted_prompt = apply_chat_template(
                processor, config, prompt,
                num_images=len(image_path),
                num_audios=len(audio_path)
            )

            output = generate(
                model,
                processor,
                formatted_prompt,
                image=image_path,
                audio=audio_path,
                verbose=True
            )

            st.text_area("Model Output", output, height=300)

        except FileNotFoundError:
            st.error(f"Error: Model not found at path '{model_path}'.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload both an image and an audio file.")
