# TTS Service Setup

This document outlines the steps to set up the isolated Python environment for the Text-to-Speech (TTS) service.

## Setup Instructions

# Navigate into the tts_service directory
cd tts_service

# Create a new virtual environment named .venv_tts
python3 -m venv .venv_tts

# Activate the new TTS environment
source .venv_tts/bin/activate

# Install the specific dependencies for the TTS service
pip install -r requirements.txt

# Deactivate to return to your shell
deactivate

# Go back to the project root
cd ..

## Running the Aura Voice Assistant

The `aura.py` script runs a Streamlit application that provides a voice-based assistant.

### Prerequisites

- Ensure you have followed the setup instructions in this README to create the `tts_service` environment.
- Make sure you have the necessary models downloaded. **https://huggingface.co/surfiniaburger** (pick tts and expert model here)

### Usage

1.  **Navigate to the project's root directory.**
2.  **Activate the main Python environment** (not the `tts_service` environment).
3.  **Run the Streamlit app:**

    ```bash
    streamlit run aura.py
    ```

This will launch a web interface where you can interact with the voice assistant.