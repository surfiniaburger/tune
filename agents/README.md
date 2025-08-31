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
    uvicorn main:app --reload
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


import logging
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent

# Import the tools we created
from tools.diagnosis_tool import diagnose_plant_from_huggingface
from tools.tts_tool import generate_speech_from_text

# --- Agent Instructions ---

DIAGNOSIS_AGENT_INSTRUCTION = """
You are a specialist agent responsible for diagnosing plant health.
Your ONLY job is to use the available tool to get a diagnosis for the image located at the following path.

**Workflow:**
1.  You **MUST** call the `diagnose_plant_from_huggingface` tool.
2.  You **MUST** pass the following path to the tool's `image_path_on_server` argument: {image_path}
3.  Your final output **MUST** be the exact text result returned by the tool. Do not add any other text.
"""

TTS_AGENT_INSTRUCTION = """
You are a specialist agent responsible for converting text to speech.
Your ONLY job is to use the available tool to generate audio from text.

**Workflow:**
1.  You will receive the diagnosis text as your direct input from the previous agent.
2.  You **MUST** call the `generate_speech_from_text` tool with this text.
3.  You **MUST** take the URL of the generated audio file returned by the tool and store it in the session state key `audio_path`.
4.  Your final output should be the diagnosis text that you received as input, without any changes. This is so the final result contains both the text and the audio path (from the state).
"""

def create_aura_mind_agent_system() -> SequentialAgent:
    """
    Builds and returns the complete Aura-Mind agent system.
    This version is simplified to be more robust and efficient by making the
    SequentialAgent the top-level agent, removing the need for a RootAgent.
    """
    logging.info("Creating simplified Aura-Mind agent system...")

    # 1. Define the specialist agents
    # The ADK will automatically use the GOOGLE_API_KEY environment variable for authentication.
    diagnosis_agent = LlmAgent(
        name="DiagnosisAgent",
        model="gemini-2.5-flash",
        instruction=DIAGNOSIS_AGENT_INSTRUCTION,
        tools=[diagnose_plant_from_huggingface],
    )

    tts_agent = LlmAgent(
        name="TtsAgent",
        model="gemini-2.5-flash",
        instruction=TTS_AGENT_INSTRUCTION,
        tools=[generate_speech_from_text],
    )

    # 2. Define the orchestrator which is now our top-level agent
    aura_mind_orchestrator_agent = SequentialAgent(
        name="AuraMindOrchestratorAgent",
        sub_agents=[diagnosis_agent, tts_agent],
        description="Orchestrates the two-step process of diagnosing a plant image and generating speech for the result."
    )

    logging.info("Aura-Mind agent system created successfully.")
    return aura_mind_orchestrator_agent


import logging
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent

# Import the tools we created
from tools.diagnosis_tool import diagnose_plant_from_huggingface
from tools.tts_tool import generate_speech_from_text

# --- Agent Instructions ---

DIAGNOSIS_AGENT_INSTRUCTION = """
You are a specialist agent responsible for diagnosing plant health.
Your ONLY job is to use the available tool to get a diagnosis for an image.

**Workflow:**
1.  You will receive the path to an image in the session state key `image_path`.
2.  You **MUST** call the `diagnose_plant_from_huggingface` tool with this image path.
3.  Your final output **MUST** be the exact text result returned by the tool. Do not add any other text.
"""

TTS_AGENT_INSTRUCTION = """
You are a specialist agent responsible for converting text to speech.
Your ONLY job is to use the available tool to generate audio from text.

**Workflow:**
1.  You will receive the diagnosis text as your direct input from the previous agent.
2.  You **MUST** call the `generate_speech_from_text` tool with this text.
3.  You **MUST** take the URL of the generated audio file returned by the tool and store it in the session state key `audio_path`.
4.  Your final output should be the diagnosis text that you received as input, without any changes. This is so the final result contains both the text and the audio path (from the state).
"""

def create_aura_mind_agent_system() -> SequentialAgent:
    """
    Builds and returns the complete Aura-Mind agent system.
    This version is simplified to be more robust and efficient by making the
    SequentialAgent the top-level agent, removing the need for a RootAgent.
    """
    logging.info("Creating simplified Aura-Mind agent system...")

    # 1. Define the specialist agents
    # The ADK will automatically use the GOOGLE_API_KEY environment variable for authentication.
    diagnosis_agent = LlmAgent(
        name="DiagnosisAgent",
        model="gemini-2.5-flash",
        instruction=DIAGNOSIS_AGENT_INSTRUCTION,
        tools=[diagnose_plant_from_huggingface],
    )

    tts_agent = LlmAgent(
        name="TtsAgent",
        model="gemini-2.5-flash",
        instruction=TTS_AGENT_INSTRUCTION,
        tools=[generate_speech_from_text],
    )

    # 2. Define the orchestrator which is now our top-level agent
    aura_mind_orchestrator_agent = SequentialAgent(
        name="AuraMindOrchestratorAgent",
        sub_agents=[diagnosis_agent, tts_agent],
        description="Orchestrates the two-step process of diagnosing a plant image and generating speech for the result."
    )

    logging.info("Aura-Mind agent system created successfully.")
    return aura_mind_orchestrator_agent
