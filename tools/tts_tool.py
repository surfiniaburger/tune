import os
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_speech_from_text(text_to_speak: str) -> str | None:
    """
    Generates speech from text by calling the TTS script in a separate environment.

    Args:
        text_to_speak: The text to be converted to speech.

    Returns:
        The path to the generated audio file, or None if an error occurs.
    """
    logging.info(f"Attempting to generate speech for text: '{text_to_speak[:50]}...'")
    try:
        # Get the absolute path to the project directory for robust pathing
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Go up one level from /tools

        # In the Docker container or local setup, the TTS virtual environment is at a fixed path.
        # This path needs to be created by the user as per the original README.
        tts_env_python = os.path.join(project_root, "tts_service/.venv_tts/bin/python")
        tts_script = os.path.join(project_root, "tts_service/run_tts_service.py")

        # IMPORTANT: This path must point to where the user has downloaded the TTS model.
        tts_model_path = os.path.join(project_root, "orpheus-3b-pidgin-voice-v1")

        # Define a predictable output path for the generated speech.
        speech_output_path = os.path.join(project_root, "static/generated_speech.wav")

        # Ensure the static directory exists
        os.makedirs(os.path.dirname(speech_output_path), exist_ok=True)

        # Check for necessary files and directories
        if not os.path.exists(tts_env_python):
            error_msg = f"TTS virtual environment not found at {tts_env_python}. Please run the setup instructions from the original README."
            logging.error(error_msg)
            return f"Error: {error_msg}"

        if not os.path.exists(tts_model_path):
            error_msg = f"TTS model not found at path: {tts_model_path}. Please download it as per the original README."
            logging.error(error_msg)
            return f"Error: {error_msg}"

        # Sanitize the text for the TTS model by replacing newlines with spaces.
        sanitized_tts_text = text_to_speak.replace('\n', ' ').strip()

        # --- Call the TTS script in the separate environment ---
        command = [
            tts_env_python,
            tts_script,
            "--text", sanitized_tts_text,
            "--model-path", tts_model_path,
            "--output-path", speech_output_path
        ]

        logging.info(f"Running TTS command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            # The TTS script from the original repo appends `_000` to the filename.
            base, ext = os.path.splitext(speech_output_path)
            actual_speech_path = f"{base}_000{ext}"

            if os.path.exists(actual_speech_path):
                # The tool should return the web-accessible URL path, not the filesystem path.
                web_path = "/" + os.path.relpath(actual_speech_path, os.path.dirname(project_root)).replace(os.sep, '/')
                logging.info(f"Speech generated successfully at {actual_speech_path}. Web path: {web_path}")
                return web_path
            else:
                logging.error(f"Generated speech file not found. The TTS script might have failed silently. Stderr: {result.stderr}")
                return "Error: TTS script ran but the output file was not found."
        else:
            logging.error(f"An error occurred during speech generation. Stderr: {result.stderr}")
            return f"Error during speech generation: {result.stderr}"

    except Exception as e:
        logging.error(f"An unexpected error occurred in generate_speech_from_text: {e}", exc_info=True)
        return f"An unexpected error occurred in the TTS tool: {e}"

if __name__ == '__main__':
    # Example usage:
    text = "Hello, this is a test of the text to speech service."
    audio_url = generate_speech_from_text(text)
    if audio_url:
        print(f"\nSpeech generated successfully. Audio URL: {audio_url}")
    else:
        print("\nFailed to generate speech.")
