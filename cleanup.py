"""
==============================================================================
CLEANUP.PY
==============================================================================
This script handles cleaning up artifacts from a completed sweep run to conserve
disk space on Kaggle.

It is called by `run_sweep.py` after evaluation is finished for a run.

It intelligently removes the largest, non-essential files:
- Deletes the `checkpoint-*` directories, which contain large optimizer states
  and intermediate model weights.
- KEEPS the final adapter files and the `evaluation_results.json`.
"""

import os
import json
import shutil
import glob

CONFIG_PATH = "current_run_config.json"

def main():
    print("--- Starting Cleanup --- ")

    # Load run configuration to find the correct output directory
    if not os.path.exists(CONFIG_PATH):
        print(f"Warning: Could not find {CONFIG_PATH}. Nothing to clean.")
        return

    with open(CONFIG_PATH, 'r') as f:
        run_config = json.load(f)
    
    run_name = run_config.get("run_name")
    if not run_name:
        print("Warning: `run_name` not found in config. Cannot clean up.")
        return

    output_dir = f"outputs/{run_name}"
    if not os.path.isdir(output_dir):
        print(f"Warning: Output directory {output_dir} not found. Nothing to clean.")
        return

    # Find all checkpoint directories within the run's output folder
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))

    if not checkpoint_dirs:
        print("No checkpoint directories found to clean.")
    else:
        for ckpt_dir in checkpoint_dirs:
            if os.path.isdir(ckpt_dir):
                try:
                    print(f"Deleting checkpoint directory: {ckpt_dir}")
                    shutil.rmtree(ckpt_dir)
                    print(f"âœ… Successfully removed {ckpt_dir}")
                except OSError as e:
                    print(f"Error removing directory {ckpt_dir}: {e}")

    print("--- Cleanup Complete ---")

if __name__ == "__main__":
    main()