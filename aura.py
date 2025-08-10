import streamlit as st
import os
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import subprocess
import numpy as np

# --- Performance Tracking Setup ---
# Initialize session state for storing performance metrics if it doesn't exist.
if 'vlm_performance_data' not in st.session_state:
    st.session_state.vlm_performance_data = []
if 'tts_performance_data' not in st.session_state:
    st.session_state.tts_performance_data = []

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

        # --- Capture VLM Performance ---
        vlm_stats = {
            "Prompt Tokens": output.prompt_tokens,
            "Generation Tokens": output.generation_tokens,
            "Prompt TPS": output.prompt_tps,
            "Generation TPS": output.generation_tps,
            "Peak Memory (GB)": output.peak_memory
        }
        st.session_state.vlm_performance_data.append(vlm_stats)


        st.markdown("### Model Output")
        st.write(output.text)

        # --- Semantic Search and TTS Preparation ---
        import faiss
        from sentence_transformers import SentenceTransformer

        def search(query, model_name='all-MiniLM-L6-v2', index_path='faiss_index.bin', data_path='documents.npy', k=1):
            """Searches the FAISS index for the most similar documents to a query."""
            model = SentenceTransformer(model_name)
            index = faiss.read_index(index_path)
            documents = np.load(data_path, allow_pickle=True)
            
            query_embedding = model.encode([query], convert_to_tensor=False)
            distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:
                    results.append({
                        'distance': distances[0][i],
                        'document': documents[idx]
                    })
            return results

        query = output.text.strip()
        search_results = search(query)

        rag_text_for_display = None
        tts_text = query  # Default to VLM output if no remedy is found

        if search_results:
            st.markdown("### Recommended Actions")
            rag_text_for_display = search_results[0]['document']
            
            # For TTS, use the first two paragraphs as a summary.
            paragraphs = rag_text_for_display.split('\n\n')
            tts_text = "\n\n".join(paragraphs[:2])

            st.markdown(rag_text_for_display)
        else:
            st.warning("No relevant information found.")

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

                # --- Capture TTS Performance ---
                # Extract performance metrics from the TTS script's stdout.
                tts_log = result.stdout
                if tts_log:
                    try:
                        # Example of parsing: "Generation Speed: 123.45 tokens/sec"
                        speed_line = [line for line in tts_log.split('\n') if "Generation Speed" in line]
                        if speed_line:
                            tts_speed = float(speed_line[0].split(':')[1].strip().split()[0])
                            st.session_state.tts_performance_data.append({"Generation Speed (tokens/sec)": tts_speed})
                    except (IndexError, ValueError) as e:
                        st.warning(f"Could not parse TTS performance data: {e}")


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

# --- Performance Dashboard ---
st.sidebar.title("On-Device Performance Dashboard")

if st.session_state.vlm_performance_data:
    st.sidebar.markdown("### Vision & Language Model (VLM) Performance")
    vlm_df = pd.DataFrame(st.session_state.vlm_performance_data)
    st.sidebar.dataframe(vlm_df)
    
    st.sidebar.markdown("**VLM Performance Over Time**")
    st.sidebar.line_chart(vlm_df[["Prompt TPS", "Generation TPS"]])
    st.sidebar.line_chart(vlm_df[["Peak Memory (GB)"]])

if st.session_state.tts_performance_data:
    st.sidebar.markdown("### Text-to-Speech (TTS) Performance")
    tts_df = pd.DataFrame(st.session_state.tts_performance_data)
    st.sidebar.dataframe(tts_df)

    st.sidebar.markdown("**TTS Performance Over Time**")
    st.sidebar.line_chart(tts_df)

if st.sidebar.button("Clear Performance Data"):
    st.session_state.vlm_performance_data = []
    st.session_state.tts_performance_data = []
    st.rerun()
