import streamlit as st
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import subprocess
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
    import gc
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model_path = "./finetuned_model_for_conversion"
    try:
        model, processor = load(model_path)
        config = model.config

        prompt = "Classify the condition of the maize plant. Choose from Healthy Maize Plant, Maize Phosphorus Deficiency."
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
            max_tokens=20,
            verbose=True
        )

        st.markdown("### Model Output")
        st.write(output.text)

        # --- RAG and TTS Preparation ---
        # Based on the classification, retrieve and display relevant information.
        # A summary is prepared for TTS to keep the audio concise.
        rag_text_for_display = None
        tts_text = output.text  # Default to VLM output if no remedy is found

        classification_result = output.text.strip()
        remedy_file_path = None

        if "Healthy Maize Plant" in classification_result:
            remedy_file_path = "healthy_maize_remedy.txt"
        elif "Maize Phosphorus Deficiency" in classification_result:
            remedy_file_path = "maize_phosphorus_deficiency_remedy.txt"
        else:
            remedy_file_path = "comic_relief.txt"

        if remedy_file_path:
            st.markdown("### Recommended Actions")
            try:
                with open(remedy_file_path, 'r', encoding='utf-8') as f:
                    rag_text_for_display = f.read()

                # For TTS, use the first two paragraphs as a summary.
                paragraphs = rag_text_for_display.split('\n\n')
                tts_text = "\n\n".join(paragraphs[:2])

                st.markdown(rag_text_for_display)
            except FileNotFoundError:
                st.warning(f"Remedy file not found: {remedy_file_path}")

        # --- Memory Cleanup ---
        # Explicitly delete the large vision model and processor to free up
        # memory before loading the TTS model. This is crucial on systems
        # with limited RAM to prevent crashes.
        with st.spinner("Clearing vision model from memory..."):
            del model
            del processor
            gc.collect()

        # --- Text-to-Speech Generation ---
        st.markdown("### Generated Speech")
        try:
            # Get the absolute path to the project directory for robust pathing
            project_root = os.path.dirname(os.path.abspath(__file__))
            tts_env_python = os.path.join(project_root, "tts_service", ".venv_tts", "bin", "python")
            tts_script = os.path.join(project_root, "tts_service", "run_tts_service.py")

            # IMPORTANT: Replace with the actual path to your downloaded model
            # Make model path absolute to avoid ambiguity in the subprocess.
            tts_model_path = os.path.join(project_root, "orpheus-3b-pidgin-voice-v1")
            
            # Check if the model path exists
            if not os.path.exists(tts_env_python):
                st.error("TTS virtual environment not found. Please run the setup instructions in Step 3.")
            elif not os.path.exists(tts_model_path):
                st.error(f"TTS model not found at path: {tts_model_path}")
                st.info("Please run `python3 download_model.py` to download the TTS model.")
            else:
                # Make output path absolute to ensure we know where to find it.
                speech_output_path = os.path.join(project_root, "generated_speech.wav")

                # Sanitize the text for the TTS model by replacing newlines with spaces.
                # This prevents errors with models that can't handle multi-line input.
                sanitized_tts_text = tts_text.replace('\n', ' ')

                # --- Call the TTS script in the separate environment ---
                command = [
                    tts_env_python,
                    tts_script,
                    "--text", sanitized_tts_text,
                    "--model-path", tts_model_path,
                    "--output-path", speech_output_path
                ]
                with st.spinner("Generating speech..."):
                    result = subprocess.run(command, capture_output=True, text=True, check=False)

                if result.returncode == 0:
                    # The TTS script appends `_000` to the filename. We need to account for that.
                    base, ext = os.path.splitext(speech_output_path)
                    actual_speech_path = f"{base}_000{ext}"

                    # Check if the file was actually created before trying to open it.
                    if os.path.exists(actual_speech_path):
                        # Read the generated audio file into a bytes object
                        # to prevent race conditions with Streamlit's file handling.
                        with open(actual_speech_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format="audio/wav")
                        st.success("Speech generated successfully!")
                        if result.stdout:
                            with st.expander("See TTS Log"):
                                st.code(result.stdout)
                    else:
                        st.error("Generated speech file not found. The TTS script might have failed silently.")
                        st.code(f"Expected file at: {actual_speech_path}")
                        st.code(f"TTS Service stdout:\n{result.stdout}")
                        st.code(f"TTS Service stderr:\n{result.stderr}")
                else:
                    st.error("An error occurred during speech generation.")
                    st.code(f"TTS Service Error:\n{result.stderr}")

        except Exception as e:
            st.error(f"An error occurred during speech generation: {e}")
    except FileNotFoundError:
        st.error(f"Error: Model not found at path '{model_path}'.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please record audio and provide an image to run the model.")
