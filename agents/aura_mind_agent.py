import logging
import os
from dotenv import load_dotenv
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
3.  You **MUST** take the text result returned by the tool and store it in the session state key `diagnosis_text`.
4.  Your final output should be a simple confirmation message, like "Diagnosis complete."
"""

TTS_AGENT_INSTRUCTION = """
You are a specialist agent responsible for converting text to speech.
Your ONLY job is to use the available tool to generate audio from text.

**Workflow:**
1.  You will receive the diagnosis text in the session state key `diagnosis_text`.
2.  You **MUST** call the `generate_speech_from_text` tool with this text.
3.  You **MUST** take the URL of the generated audio file returned by the tool and store it in the session state key `audio_path`.
4.  Your final output should be a simple confirmation message, like "Speech generation complete."
"""

def create_aura_mind_agent_system() -> SequentialAgent:
    """
    Builds and returns the complete Aura-Mind agent system.
    """
    logging.info("Creating Aura-Mind agent system...")

    # Load environment variables from .env file
    load_dotenv()
    API_KEY = os.environ.get("GOOGLE_API_KEY")

    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY must be set in your .env file.")

    # Shared config for all LlmAgents to use the Google AI API Key
    google_ai_config = {
        "api_key": API_KEY,
    }

    # 1. Define the specialist agents
    diagnosis_agent = LlmAgent(
        name="DiagnosisAgent",
        model="gemini-2.5-flash", # A small, fast model is sufficient for this
        instruction=DIAGNOSIS_AGENT_INSTRUCTION,
        tools=[diagnose_plant_from_huggingface],
        output_key="diagnosis_confirmation", # We store the main result in session state
        **google_ai_config
    )

    tts_agent = LlmAgent(
        name="TtsAgent",
        model="gemini-2.5-flash",
        instruction=TTS_AGENT_INSTRUCTION,
        tools=[generate_speech_from_text],
        output_key="tts_confirmation",
        **google_ai_config
    )

    # 2. Define the orchestrator (factory foreman)
    aura_mind_orchestrator_agent = SequentialAgent(
        name="AuraMindOrchestratorAgent",
        sub_agents=[diagnosis_agent, tts_agent],
        description="Orchestrates the two-step process of diagnosing a plant image and generating speech for the result."
    )

    logging.info("Aura-Mind agent system created successfully.")
    return aura_mind_orchestrator_agent
