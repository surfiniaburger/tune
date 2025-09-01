"""
==============================================================================
TRAIN.PY (v9 - Final Simplification Fix for Wrapper)
==============================================================================
This script is configured for multi-GPU training with OpenSloth.

v9 FIXES:
- The `opensloth` wrapper is failing to correctly handle either the `quantization_config`
  object or the individual `bnb_*` kwargs.
- This version simplifies the FastModelArgs to the most basic arguments that the
  wrapper is designed to handle: `load_in_4bit` and `dtype`.
- By providing only the simplest, most essential flags, we allow the underlying,
  up-to-date Unsloth library to take over and correctly infer the necessary
  quantization settings.
"""

import os

# --- set environment variables before importing Unsloth ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["UNSLOTH_DISABLE_FUSED_LOSS"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # extra safety

import json
import torch
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
        raise FileNotFoundError(f"Could not find {config_path}. This script should be called by run_sweep.py.")
    with open(config_path, 'r') as f:
        print(f"Loading configuration from {config_path}")
        return json.load(f)

def main():
    run_config = load_run_config()
    run_name = run_config["run_name"]

    # --- 2. Configure OpenSloth and Training Arguments for Multi-GPU ---
    DEVICES, GLOBAL_BZ, BZ = [0, 1], 16, run_config["per_device_train_batch_size"]
    if GLOBAL_BZ % (len(DEVICES) * BZ) != 0: raise ValueError("GLOBAL_BZ must be divisible by (num_devices * per_device_batch_size)")
    GRAD_ACCUM = GLOBAL_BZ // (len(DEVICES) * BZ)

    opensloth_config = OpenSlothConfig(
        data_cache_path="data/cached_train_dataset_hf_augmented",
        devices=DEVICES,
        fast_model_args=FastModelArgs(
            model_name="unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None, 
            # -----------------------------

            use_gradient_checkpointing="unsloth",
        ),
        lora_args=LoraArgs(
            r=run_config["lora_r"], lora_alpha=run_config["lora_alpha"],
            target_modules=[ "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
            lora_dropout=run_config["lora_dropout"], bias="none", use_rslora=False,
            finetune_vision_layers=False, finetune_language_layers=True,
        ),
        sequence_packing=False,
    )

    training_config = TrainingArguments(
        output_dir="outputs/vision_multiGPU_experiment",
        per_device_train_batch_size=BZ, gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=run_config["learning_rate"], num_train_epochs=run_config["num_train_epochs"],
        weight_decay=run_config["weight_decay"], logging_steps=10, lr_scheduler_type="linear",
        max_grad_norm=1, warmup_ratio=0.1, save_total_limit=1, save_steps=100,
        optim="adamw_torch_fused", seed=3407, remove_unused_columns=False,
        dataset_text_field="", max_seq_length=1024, dataloader_pin_memory=True,
        fp16=True, report_to="wandb", resume_from_checkpoint="", torch_compile=False,
    )

    # --- 3. Setup W&B and Run Training ---
    os.environ["WANDB_PROJECT"] = "open-maize-vision-sweep"
    os.environ["WANDB_NAME"] = run_name

    print(f"--- Starting Training for: {run_name} (OpenSloth Multi-GPU) ---")
    print(f"  Learning Rate: {run_config['learning_rate']}, Epochs: {run_config['num_train_epochs']}, LoRA R: {run_config['lora_r']}")
    print(f"  Global batch size: {len(DEVICES) * BZ * GRAD_ACCUM}"); print(f"  Gradient accumulation steps: {GRAD_ACCUM}")

    setup_envs(opensloth_config, training_config)
    run_mp_training(opensloth_config.devices, opensloth_config, training_config)

    print(f"--- Finished Training for: {run_name} ---")

if __name__ == "__main__":
    main()