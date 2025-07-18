# download_model.py
from huggingface_hub import snapshot_download

MODEL_ID = "mlx-community/gemma-3n-E2B-it-4bit"

print(f"Downloading model: {MODEL_ID}")
snapshot_download(repo_id=MODEL_ID)
print("✅ Download complete!")