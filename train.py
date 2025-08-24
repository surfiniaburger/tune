"""
==============================================================================
TRAIN.PY (v3 - OpenSloth Multi-GPU)
==============================================================================
This script is configured to use the internal multi-GPU handling of opensloth.
It is called directly via `python train.py`, and the `run_mp_training` function
manages the process spawning and training across multiple devices.
"""

import os
import json
from opensloth.opensloth_config import (
    FastModelArgs,
    LoraArgs,
    OpenSlothConfig,
    TrainingArguments,
)
from opensloth.scripts.opensloth_sft_trainer import run_mp_training, setup_envs

# --- 1. Load Configuration for the Current Run ---
def load_run_config():
    config_path = 'current_run_config.json'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find {config_path}. "
                              f"This script should be called by run_sweep.py.")
    with open(config_path, 'r') as f:
        print(f"Loading configuration from {config_path}")
        return json.load(f)

def main():

    run_config = load_run_config()
    run_name = run_config["run_name"]
    output_dir = f"outputs/{run_name}"

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # The directory creation is handled by the main process inside opensloth
    if os.environ.get('RANK') == '0':
        os.makedirs(output_dir, exist_ok=True)
    # --- 2. Configure OpenSloth and Training Arguments for Multi-GPU ---

    # Multi-GPU Configuration
    DEVICES = [0, 1]
    GLOBAL_BZ = 16
    BZ = run_config["per_device_train_batch_size"]
    
    if GLOBAL_BZ % (len(DEVICES) * BZ) != 0:
        raise ValueError("GLOBAL_BZ must be divisible by (num_devices * per_device_batch_size)")
    GRAD_ACCUM = GLOBAL_BZ // (len(DEVICES) * BZ)

    opensloth_config = OpenSlothConfig(
        data_cache_path="data/cached_train_dataset_hf",
        devices=DEVICES,
        fast_model_args=FastModelArgs(
            model_name="unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,
            use_gradient_checkpointing="unsloth",
        ),
        lora_args=LoraArgs(
            r=run_config["lora_r"],
            lora_alpha=run_config["lora_alpha"],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=run_config["lora_dropout"],
            bias="none",
            use_rslora=False,
            finetune_vision_layers=True,
            finetune_language_layers=True,
        ),
        sequence_packing=False,
    )

    training_config = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BZ,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=run_config["learning_rate"],
        num_train_epochs=run_config["num_train_epochs"],
        weight_decay=run_config["weight_decay"],
        logging_steps=10,
        lr_scheduler_type="linear",
        max_grad_norm=1,
        warmup_ratio=0.1,
        save_total_limit=1,
        save_steps=100,
        optim="adamw_torch_fused",
        seed=3407,
        remove_unused_columns=False,
        dataset_text_field="",
        max_seq_length=1024,
        dataloader_pin_memory=True,
        fp16=True,
        report_to="wandb",
        resume_from_checkpoint="",
        torch_compile=False,
    )

    # --- 3. Setup W&B and Run Training ---
    os.environ["WANDB_PROJECT"] = "open-maize-vision-sweep"
    os.environ["WANDB_NAME"] = run_name

    print(f"--- Starting Training for: {run_name} (OpenSloth Multi-GPU) ---")
    print(f"  Output Directory: {output_dir}")
    print(f"  Learning Rate: {run_config['learning_rate']}, Epochs: {run_config['num_train_epochs']}, LoRA R: {run_config['lora_r']}")
    print(f"  Global batch size: {len(DEVICES) * BZ * GRAD_ACCUM}")
    print(f"  Gradient accumulation steps: {GRAD_ACCUM}")

    # Let opensloth manage the environment setup and multi-process training
    setup_envs(opensloth_config, training_config)
    run_mp_training(opensloth_config.devices, opensloth_config, training_config)

    print(f"--- Finished Training for: {run_name} ---")

if __name__ == "__main__":
    main()
