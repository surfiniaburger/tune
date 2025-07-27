# prepare_dataset.py (v7 - For custom training loop)
import os
import json
import random
from pathlib import Path
from tqdm import tqdm
from src.aura_mind.data_utils import CLASS_NAME_MAPPING

# --- This script now ONLY prepares the JSONL files. Augmentation will be done on-the-fly. ---

DATASET_ROOT = Path("datasets/train")
OUTPUT_DIR = Path("mlx_dataset")
VALIDATION_SPLIT_RATIO = 0.15

# In prepare_dataset.py

def create_prompt(class_name: str) -> str:
    """
    Creates the simplest possible prompt with the required image token.
    We are removing all other special tags like [INST] to avoid conflicts.
    """
    # The <image> token tells the model where to insert the image data.
    # It must be present.
    return f"What disease is this? <image>\nThis is {class_name}."

def main():
    print("🚀 Starting dataset preparation (v7 for custom training loop)...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if (OUTPUT_DIR / "train.jsonl").exists(): os.remove(OUTPUT_DIR / "train.jsonl")
    if (OUTPUT_DIR / "valid.jsonl").exists(): os.remove(OUTPUT_DIR / "valid.jsonl")
    
    all_image_paths = list(DATASET_ROOT.glob("**/*.jpg"))
    random.shuffle(all_image_paths)

    split_index = int(len(all_image_paths) * (1 - VALIDATION_SPLIT_RATIO))
    train_paths = all_image_paths[:split_index]
    valid_paths = all_image_paths[split_index:]

    print(f"Total images: {len(all_image_paths)}, Train: {len(train_paths)}, Valid: {len(valid_paths)}")

    def write_jsonl(paths, output_file):
        with open(output_file, "w") as f_out:
            for image_path in tqdm(paths, desc=f"Writing {output_file.name}"):
                class_folder_name = image_path.parent.name
                if class_folder_name not in CLASS_NAME_MAPPING: continue
                
                display_name = CLASS_NAME_MAPPING[class_folder_name]
                prompt_text = create_prompt(display_name)
                
                record = {"image": str(image_path), "text": prompt_text}
                f_out.write(json.dumps(record) + "\n")

    write_jsonl(train_paths, OUTPUT_DIR / "train.jsonl")
    write_jsonl(valid_paths, OUTPUT_DIR / "valid.jsonl")

    print(f"\n✅ Dataset preparation complete! Files are in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()