# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment for the main app
RUN python -m venv /app/venv_main
ENV PATH="/app/venv_main/bin:$PATH"

# Install main app dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir \
        "mlx-vlm>=0.3.2" \
        "mlx>=0.28.0" \
        "mlx-lm>=0.26.3" \
        "mlx-metal>=0.28.0" \
        "cryptography" \
        "numpy" \
        "Pillow" \
        "huggingface-hub>=0.20.0" \
        "filelock>=3.13.0" \
        "datasets" \
        "tqdm" \
        "transformers" \
        "kaggle" \
        "streamlit" \
        "gradio" \
        "sentence-transformers" \
        "PyMuPDF" \
        "numba>=0.59.0" \
        "llvmlite>=0.42.0" \
        "faiss-cpu"

# Create and activate virtual environment for the TTS service
RUN python -m venv /app/venv_tts

# Install TTS service dependencies
COPY tts_service/requirements.txt ./tts_requirements.txt
RUN /app/venv_tts/bin/pip install --no-cache-dir -r tts_requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "aura.py"]
