# create_zip.py
import shutil
from pathlib import Path

# --- Configuration ---

# The folder we want to zip. This points to your organized training data.
SOURCE_FOLDER = Path("datasets/validation")

# The name of the final zip file we want to create.
OUTPUT_ZIP_FILE = Path("aura_mind_validation_dataset") # .zip will be added automatically

def main():
    """
    Finds the source folder and compresses it into a single .zip file.
    """
    print("🚀 Starting dataset compression...")

    if not SOURCE_FOLDER.is_dir():
        print(f"❌ Error: Source folder not found at '{SOURCE_FOLDER}'")
        print("Please make sure your 'datasets/train' folder exists.")
        return

    print(f"Source folder: '{SOURCE_FOLDER}'")
    print(f"Output file:   '{OUTPUT_ZIP_FILE}.zip'")

    try:
        # The core command to create the zip archive
        # 'zip' is the format, SOURCE_FOLDER is what to zip
        shutil.make_archive(
            base_name=str(OUTPUT_ZIP_FILE),
            format='zip',
            root_dir=SOURCE_FOLDER
        )
        print(f"\n✅ Successfully created '{OUTPUT_ZIP_FILE}.zip'!")
        print("You are now ready to upload this file to your Google Drive.")

    except Exception as e:
        print(f"\n❌ An error occurred during zipping: {e}")


if __name__ == "__main__":
    main()