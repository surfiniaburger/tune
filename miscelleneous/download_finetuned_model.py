# download_finetuned_model.py
from huggingface_hub import snapshot_download
from pathlib import Path

# --- Configuration ---
# Your public Hugging Face repository ID
hf_repo_id = "surfiniaburger/AuraMind-Maize-Expert-v1"

# The local directory where the model will be saved
# We'll create it inside the main 'tune' folder
local_model_path = Path("./finetuned_model_for_conversion")

def main():
    """
    Downloads the specified Hugging Face model repository to a local directory.
    """
    print(f"🚀 Downloading your fine-tuned model '{hf_repo_id}'...")
    print(f"   Saving to: {local_model_path.resolve()}")

    # 'local_dir_use_symlinks=False' is important to avoid issues with some file systems
    # 'ignore_patterns' is used to skip the very large .safetensors file if needed,
    # but for a full conversion, we need everything.
    snapshot_download(
        repo_id=hf_repo_id,
        local_dir=local_model_path,
        local_dir_use_symlinks=False
    )
    
    print("\n✅ Your fine-tuned model has been downloaded successfully.")
    print("\n--- Directory contents ---")
    # Use os.system to run shell commands for verification
    import os
    os.system(f"ls -lh {local_model_path}")

if __name__ == "__main__":
    main()