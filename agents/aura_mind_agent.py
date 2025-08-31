import logging
from google.adk.agents.llm_agent import LlmAgent

# Import the tools we created
from tools.diagnosis_tool import diagnose_plant_from_huggingface

# --- Agent Instructions ---

AURA_MIND_AGENT_INSTRUCTION = """Your system instruction contains a path to an image. 
You MUST call the `diagnose_plant_from_huggingface` tool.
For the `image_path_on_server` argument, you MUST use the image path provided in your instruction.
The image path is: {image_path}

After the tool returns the diagnosis, your job is to output the diagnosis text exactly as you received it. Do not add any other text or formatting.
"""

def create_aura_mind_agent_system() -> LlmAgent:
    """
    Builds and returns the complete Aura-Mind agent system.
    This version uses a single, efficient LlmAgent to handle the diagnosis workflow.
    """
    logging.info("Creating diagnosis-only Aura-Mind agent system...")

    # The ADK will automatically use the GOOGLE_API_KEY environment variable for authentication.
    aura_mind_agent = LlmAgent(
        name="AuraMindAgent",
        model="gemini-2.5-flash",
        instruction=AURA_MIND_AGENT_INSTRUCTION,
        tools=[
            diagnose_plant_from_huggingface,
        ],
    )

    logging.info("Aura-Mind agent system created successfully.")
    return aura_mind_agent
