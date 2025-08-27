# Aura-Mind ADK Proof-of-Concept

This directory contains a proof-of-concept demonstrating how to refactor the Aura-Mind application using the Google Agent Development Kit (ADK). It creates a modular, agent-based system that communicates with the existing Hugging Face Space for diagnosis and uses the local TTS service for speech generation.

This implementation is based on the architecture from the `galactic-streamhub` project, using a multi-agent approach with specific, detailed instructions for each agent.

## Prerequisites

*   Python 3.11+
*   `uv` (run `pip install uv` if you don't have it)
*   The `tts_service` virtual environment must be set up as per the original project's `README.md` (in the root directory).
*   The `orpheus-3b-pidgin-voice-v1` TTS model must be downloaded and placed in the project's root directory, as per the original `README.md`.

## Setup & Installation

1.  **Configure Environment Variables:**
    The ADK agents require a Google AI API Key to function.
    ```bash
    # Copy the example .env file
    cp .env.example .env
    ```
    Now, edit the `.env` file and add your Google AI API Key.

2.  **Create and activate the main virtual environment:**
    From the root of the project directory:
    ```bash
    # Create the virtual environment
    uv venv

    # Activate it (on macOS/Linux)
    source .venv/bin/activate
    ```

3.  **Install the required packages:**
    This will install `google-adk`, `fastapi`, `uvicorn`, `gradio_client`, etc., as defined in the root `pyproject.toml`.
    ```bash
    uv pip install -e .
    ```
    *(Note: The `-e .` installs the project in editable mode, which is good practice for development.)*

## Running the Application

1.  **Start the FastAPI Application:**
    Make sure you are in the main virtual environment (`source .venv/bin/activate`). Then, run the following command from the root of the project directory:
    ```bash
    uv run uvicorn main:app --reload
    ```
    The server should start and be available at `http://127.0.0.1:8000`.

2.  **Access the Web Interface:**
    Open your web browser and navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000).

## How to Use

1.  Click the "Choose File" button to upload an image of a maize plant.
2.  Once the image is selected, the "Diagnose Plant" button will become active.
3.  Click the "Diagnose Plant" button.
4.  The status box will show the progress as the application connects to the server, sends the image, and waits for the agent to complete its work. This may take some time, especially if the Hugging Face Space is starting from a cold state.
5.  Once complete, the diagnosis text will appear, and an audio player will be loaded with the spoken remedy.
