import os
from huggingface_hub import snapshot_download

def download_model(repo_id, local_dir):
    if not os.path.exists(local_dir):
        print(f'Downloading model from {repo_id} to {local_dir}...')
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        print('Download complete.')
    else:
        print(f'Model already exists in {local_dir}.')

if __name__ == '__main__':
    # Replace with your desired Hugging Face model repo ID
    model_repo = 'surfiniaburger/orpheus-3b-pidgin-voice-v1'
    # Replace with your desired local directory to save the model
    local_model_path = './orpheus-3b-pidgin-voice-v1'

    download_model(model_repo, local_model_path)
