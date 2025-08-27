import os
import json
import asyncio
import base64
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketDisconnect

from google.genai.types import Part, Content, Blob
from google.adk.runners import Runner
from google.adk.agents import LiveRequestQueue
from google.adk.sessions.in_memory_session_service import InMemorySessionService

# Import our agent system
from agents.aura_mind_agent import create_aura_mind_agent_system

# --- Configuration & Global Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

APP_NAME = "AuraMind-ADK"
STATIC_DIR = Path("static")
TEMP_IMAGE_DIR = Path("temp_images")

# Ensure temporary directory for image uploads exists
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# Initialize ADK services
session_service = InMemorySessionService()

# --- FastAPI Application Lifespan ---
@asynccontextmanager
async def app_lifespan(app_instance: FastAPI) -> Any:
    """Handles application startup and shutdown."""
    logging.info(f"{APP_NAME} starting up...")
    yield
    logging.info(f"{APP_NAME} shutting down...")

# Instantiate FastAPI app
app = FastAPI(lifespan=app_lifespan)

# --- ADK Agent Session ---
async def start_agent_session(session_id: str, image_path: str):
    """Initializes and starts a new ADK agent session."""
    logging.info(f"Starting agent session {session_id} for image {image_path}")

    # Create a new session with the image path in its state
    session_state = {"image_path": image_path}
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=session_id, session_id=session_id, state=session_state
    )

    # Create the agent system
    agent_instance = create_aura_mind_agent_system()

    runner = Runner(
        app_name=APP_NAME,
        agent=agent_instance,
        session_service=session_service,
    )

    live_request_queue = LiveRequestQueue()

    # We are not using streaming for this PoC, so we'll run and wait for the final result.
    # We will send a single empty text content to trigger the agent.
    live_request_queue.send_content(Content(parts=[Part(text="Start diagnosis.")]))
    live_request_queue.close() # Signal that no more requests will come

    final_event = None
    async for event in runner.run_live(session=session, live_request_queue=live_request_queue):
        if event.turn_complete:
            final_event = event
            break

    return final_event

# --- WebSocket Communication Logic ---
async def handle_websocket_connection(websocket: WebSocket):
    """Manages a single client WebSocket connection."""
    await websocket.accept()
    session_id = os.urandom(16).hex()
    logging.info(f"Client #{session_id} connected via WebSocket.")

    try:
        # 1. Receive the image data from the client
        message_json = await websocket.receive_text()
        message = json.loads(message_json)
        mime_type = message.get("mime_type")
        data = message.get("data")

        if not mime_type or not mime_type.startswith("image/"):
            await websocket.send_text(json.dumps({"error": "Invalid message format. Expected an image."}))
            return

        # 2. Save the image to a temporary file
        image_bytes = base64.b64decode(data)
        file_extension = mime_type.split("/")[-1]
        temp_image_path = TEMP_IMAGE_DIR / f"{session_id}.{file_extension}"

        with open(temp_image_path, "wb") as f:
            f.write(image_bytes)
        logging.info(f"Saved uploaded image for session {session_id} to {temp_image_path}")

        # 3. Start the agent session and wait for the result
        await websocket.send_text(json.dumps({"status": "starting_diagnosis", "message": "Starting diagnosis... This may take a moment."}))

        final_event = await start_agent_session(session_id, str(temp_image_path))

        # 4. Send the final result back to the client
        if final_event and final_event.content and final_event.content.parts:
            final_response_text = final_event.content.parts[0].text
            logging.info(f"Agent for session {session_id} finished with response: {final_response_text}")
            # The agent is instructed to return a raw JSON string
            await websocket.send_text(final_response_text)
        else:
            logging.error(f"Agent for session {session_id} did not return a final response.")
            await websocket.send_text(json.dumps({"error": "Agent failed to produce a result."}))

    except WebSocketDisconnect:
        logging.info(f"Client #{session_id} disconnected.")
    except Exception as e:
        logging.error(f"Error in WebSocket endpoint for session {session_id}: {e}", exc_info=True)
        try:
            if websocket.client_state.name == 'CONNECTED':
                await websocket.send_text(json.dumps({"error": "An unexpected server error occurred."}))
        except Exception:
            pass # Suppress errors during error reporting
    finally:
        logging.info(f"Cleaning up WebSocket for session {session_id}.")
        # Clean up the temporary image file
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            logging.info(f"Removed temporary image file: {temp_image_path}")
        if websocket.client_state.name == 'CONNECTED':
            await websocket.close()

# --- FastAPI Endpoints ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for agent interaction."""
    await handle_websocket_connection(websocket)

# Serve static files (HTML, JS)
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")

@app.get("/")
async def root():
    """Serves the main index.html file."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# To run: uvicorn main:app --reload
# Remember to create a .env file if your tools need it (e.g., for API keys)
# The current PoC doesn't require a .env file but it's good practice.
