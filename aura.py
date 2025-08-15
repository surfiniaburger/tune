import streamlit as st
import os
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import subprocess
import numpy as np

from database import check_if_indexed
from create_index import create_initial_index as build_secure_index
from search import search as secure_search
from ingest_document import ingest_pdf
import streamlit as st

st.title("Aura-Mind: Your Offline AI Farming Companion")

# --- Knowledge Base Management ---
with st.sidebar:
    st.header("Knowledge Base")
    if st.button("Rebuild Initial Knowledge Base"):
        with st.spinner("Deleting old base and building new one..."):
            docs = {
                "Healthy Maize Plant": "For a Healthy Maize Plant, ensure proper watering and sunlight. No special remedy is needed. Continue good farming practices.",
                "Maize Phosphorus Deficiency": "Phosphorus deficiency in maize is characterized by stunted growth and purplish discoloration of leaves. To remedy this, apply a phosphorus-rich fertilizer like DAP (Di-Ammonium Phosphate) or bone meal to the soil. Follow package instructions for application rates."
            }
            create_initial_index(docs)
        st.success("Initial knowledge base rebuilt!")
    
    st.markdown("---")
    st.subheader("Add Your Own Knowledge")
    uploaded_pdf = st.file_uploader("Upload a PDF document", type="pdf")
    if uploaded_pdf is not None:
        # Save the uploaded file temporarily to pass its path
        temp_file_path = os.path.join(".", uploaded_pdf.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        with st.spinner(f"Ingesting '{uploaded_pdf.name}'... This may take a while for large documents."):
            ingest_pdf(temp_file_path, uploaded_pdf.name)
        
        st.success(f"Successfully added '{uploaded_pdf.name}' to your knowledge base!")
        # Clean up the temporary file
        os.remove(temp_file_path)


# Check if the index exists. If not, offer to build it.
if not check_if_indexed():
    st.warning("Local knowledge base not found. Please build it from the sidebar to enable recommendations.")
    if st.button("Build Local Knowledge Base"):
        document_files = ["healthy_maize_remedy.txt", "maize_phosphorus_deficiency_remedy.txt", "comic_relief.txt"]
        documents_content = []
        for file_path in document_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents_content.append(f.read())
            except FileNotFoundError:
                st.error(f"Required file not found: {file_path}")
        
        with st.spinner("Building secure index... This may take a moment."):
            build_secure_index(documents_content)
        st.success("Secure knowledge base built successfully!")
        st.rerun()

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
image_path = None

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
        )

        output = generate(
            model,
            processor,
            formatted_prompt,
            image=[image_path],
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


        st.markdown("### Diagnosis")
        st.write(output.text)

        query = output.text.strip()
        search_results = secure_search(query, k=3)

        rag_text_for_display = None
        tts_text = query  # Default to VLM output if no remedy is found

        if search_results:
            for result in search_results:
                st.markdown("### Recommended Actions")
                if result['type'] == 'text':
                    st.markdown(result['content'])
                    st.caption(f"Source: Text from page {result['page']}")
                elif result['type'] == 'image':
                    st.image(result['content'], caption=f"Source: Image from page {result['page']}")
        else:
            st.warning("No relevant information found in your local knowledge base.")


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
            # In the Docker container, the TTS virtual environment is at a fixed path.
            tts_env_python = "/app/venv_tts/bin/python"
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
