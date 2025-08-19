import os
import zipfile
import tqdm  # A library for nice progress bars, run: pip install tqdm

def zip_directory(folder_path, output_path):
    """
    Zips the entire contents of a folder.

    Args:
        folder_path (str): The path to the folder to be zipped.
        output_path (str): The path for the output .zip file.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: The directory '{folder_path}' does not exist.")
        return

    print(f"Starting to zip '{folder_path}' into '{output_path}'...")

    # Get the total number of files to zip for the progress bar
    total_files = sum([len(files) for r, d, files in os.walk(folder_path)])

    try:
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Use tqdm for a progress bar
            with tqdm.tqdm(total=total_files, unit="file", desc="Zipping files") as pbar:
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        # Create the full path to the file
                        file_path = os.path.join(root, file)
                        # Create the relative path for the file inside the zip
                        arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                        zipf.write(file_path, arcname)
                        pbar.update(1)

        print(f"\n--- ✅ Success! ---")
        print(f"Folder successfully zipped to: {output_path}")

    except Exception as e:
        print(f"\n[Error] An error occurred while creating the zip file: {e}")


# --- Configuration ---

# 1. The name of the folder you want to zip.
folder_to_zip = 'maize_dataset'

# 2. The name for the output zip file.
output_zip_file = 'maize_dataset.zip'


# --- Run the Script ---
zip_directory(folder_to_zip, output_zip_file)