import logging
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools.agent_tool import AgentTool

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

ROOT_AGENT_INSTRUCTION = """
You are AVA (Advanced Visual Assistant), the primary interface for the Aura-Mind system.
Your goal is to orchestrate the diagnosis and speech generation workflow.

**Workflow:**
1.  The user has provided an image. The path to this image is in the session state key `image_path`.
2.  You **MUST** delegate the entire task to the `AuraMindOrchestratorAgent` tool. Call this tool without any arguments.
3.  After the `AuraMindOrchestratorAgent` has finished, the diagnosis and audio path will be in the session state.
4.  Your final response to the user **MUST** be a single, raw JSON object containing the diagnosis text and the audio URL.
    Example: `{"diagnosis": "The plant appears to be healthy.", "audio_url": "/static/generated_speech_000.wav"}`
    Do not add any other text or formatting.
"""


def create_aura_mind_agent_system() -> LlmAgent:
    """
    Builds and returns the complete Aura-Mind agent system.
    """
    logging.info("Creating Aura-Mind agent system...")

    # 1. Define the specialist agents
    diagnosis_agent = LlmAgent(
        name="DiagnosisAgent",
        model="gemini-2.5-flash", # A small, fast model is sufficient for this
        instruction=DIAGNOSIS_AGENT_INSTRUCTION,
        tools=[diagnose_plant_from_huggingface],
        output_key="diagnosis_confirmation" # We store the main result in session state
    )

    tts_agent = LlmAgent(
        name="TtsAgent",
        model="gemini-2.5-flash",
        instruction=TTS_AGENT_INSTRUCTION,
        tools=[generate_speech_from_text],
        output_key="tts_confirmation"
    )

    # 2. Define the orchestrator (factory foreman)
    aura_mind_orchestrator_agent = SequentialAgent(
        name="AuraMindOrchestratorAgent",
        sub_agents=[diagnosis_agent, tts_agent],
        description="Orchestrates the two-step process of diagnosing a plant image and generating speech for the result."
    )

    # 3. Wrap the orchestrator in an AgentTool so the RootAgent can call it
    orchestrator_tool = AgentTool(agent=aura_mind_orchestrator_agent)
    # Patch the tool for ADK compatibility, as seen in galactic-streamhub
    if hasattr(orchestrator_tool, 'run_async') and callable(getattr(orchestrator_tool, 'run_async')):
        orchestrator_tool.func = orchestrator_tool.run_async
        logging.info(f"Patched AgentTool '{orchestrator_tool.name}' with .func attribute.")

    # 4. Define the Root Agent (main interface)
    root_agent = LlmAgent(
        name="RootAgent",
        model="gemini-live-2.5-flash-preview",
        instruction=ROOT_AGENT_INSTRUCTION,
        tools=[orchestrator_tool]
    )

    logging.info("Aura-Mind agent system created successfully.")
    return root_agent
