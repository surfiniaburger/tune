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
