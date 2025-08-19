from datasets import load_dataset
import os

# --- Configuration ---

# 1. The name of the folder containing your dataset.
local_dataset_path = "maize_dataset"

# 2. Your Hugging Face username.
hf_username = "surfiniaburger"

# 3. The desired name for your dataset on the Hub.
#    (e.g., "maize-leaf-health", "maize-phosphorus-deficiency", etc.)
hf_dataset_name = "maize-plant-conditions"


# --- Script Logic (No need to edit below this line) ---

print("--- Starting Dataset Upload to Hugging Face Hub ---")

# Check if the dataset directory exists
if not os.path.isdir(local_dataset_path):
    print(f"\n[Error] The directory '{local_dataset_path}' was not found.")
    print("Please make sure this script is in the same directory as your 'maize_dataset' folder.")
else:
    try:
        # Load the dataset from your local folder using the "imagefolder" feature.
        # This automatically finds and uses your `metadata.csv` file.
        print(f"\nStep 1: Loading dataset from '{local_dataset_path}'...")
        dataset = load_dataset("imagefolder", data_dir=local_dataset_path)

        print("\nDataset loaded successfully!")
        print(dataset) # This will show the structure, e.g., {'train': Dataset(...)}

        # Define the full repository name on the Hub
        repo_id = f"{hf_username}/{hf_dataset_name}"

        # Upload the dataset to the Hub.
        # This will create a new public dataset repository on your profile.
        # To create a private dataset, add `private=True` to the function call.
        # e.g., dataset.push_to_hub(repo_id, private=True)
        print(f"\nStep 2: Uploading dataset to the Hub at '{repo_id}'...")
        dataset.push_to_hub(repo_id)

        print("\n--- ✅ Success! ---")
        print("Your dataset is now live on the Hugging Face Hub.")
        print(f"You can view it at: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"\n[Error] An unexpected error occurred: {e}")
        print("\nPlease double-check the following:")
        print("1. Are you logged in? (Did you run `huggingface-cli login`?)")
        print("2. Is your `local_dataset_path` spelled correctly?")
        print("3. Does your user have 'write' permissions on Hugging Face?")