"""
==============================================================================
RUN_SWEEP.PY (v3)
==============================================================================
This is the master script to run the entire hyperparameter sweep.

(v3 Update): Calls `train.py` directly without `torchrun`, allowing the
`opensloth` library to manage multi-GPU processing internally. This is the
correct way to run the training script.

Execute this script from your Kaggle notebook using: !python run_sweep.py
"""

import os
import json
import subprocess
import csv
from sweep_config import sweep_config

# --- 1. W&B Login ---
def login_to_wandb():
    """Logs into Weights & Biases using Kaggle secrets."""
    try:
        from kaggle_secrets import UserSecretsClient
        import wandb
        user_secrets = UserSecretsClient()
        wandb_api_key = user_secrets.get_secret("wandb_api_key")
        wandb.login(key=wandb_api_key)
        print("✅ Successfully logged into Weights & Biases.")
    except ImportError:
        print("Could not import UserSecretsClient. Assuming W&B is configured globally.")
    except Exception as e:
        print(f"Could not log into W&B. Please ensure the 'wandb_api_key' secret is set.")
        print(f"Error: {e}")
        exit(1)

# --- 2. Function to run a command ---
def run_command(command):
    """Executes a shell command and checks for errors."""
    print(f"\n--- Running command: {' '.join(command)} ---")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            # W&B and other libraries sometimes write info to stderr, so we print it
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise

# --- 3. Main Sweep Orchestrator ---
def main():
    login_to_wandb()
    
    print("--- Starting Data Caching Phase ---")
    run_command(["python", "cache_datasets.py", "--dataset-type", "train"])
    run_command(["python", "cache_datasets.py", "--dataset-type", "validation"])
    print("--- Data Caching Phase Complete ---")

    all_results = []
    
    for i, run_params in enumerate(sweep_config):
        run_name = run_params.get("run_name", f"run_{i+1}")
        print(f"\n=================================================================")
        print(f"Starting Sweep Run {i+1}/{len(sweep_config)}: {run_name}")
        print(f"=================================================================")

        with open("current_run_config.json", 'w') as f:
            json.dump(run_params, f, indent=4)

        try:
            # --- Create output directory on the main process ---
            output_dir = f"outputs/{run_name}"
            print(f"Ensuring output directory exists: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
            # --- CORRECTED TRAINING COMMAND ---
            # Call train.py directly. OpenSloth will handle the multi-GPU logic.
            run_command(["python", "train.py"])
            
            run_command(["python", "evaluate.py"])
            run_command(["python", "cleanup.py"])

            results_path = f"outputs/{run_name}/evaluation_results.json"
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    all_results.append(json.load(f))
                print(f"✅ Successfully completed and logged run: {run_name}")
            else:
                print(f"❌ Warning: Could not find results file for run: {run_name}")

        except Exception as e:
            print(f"❌❌❌ An error occurred during run: {run_name}. Halting sweep. ❌❌❌")
            print(f"Error: {e}")
            break

    # --- 4. Save Final Results ---
    if not all_results:
        print("\nNo results were collected. Skipping final CSV.")
        return

    print("\n--- Sweep Finished. Compiling final results... ---")
    csv_data = []
    for result in all_results:
        flat_row = result['run_config'].copy()
        flat_row['accuracy'] = result.get('accuracy')
        flat_row['correct_predictions'] = result.get('correct_predictions')
        flat_row['total_samples'] = result.get('total_samples')
        csv_data.append(flat_row)

    headers = list(csv_data[0].keys())
    
    results_csv_path = "sweep_results.csv"
    with open(results_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"✅ All sweep results saved to {results_csv_path}")

if __name__ == "__main__":
    main()
