"""
==============================================================================
SWEEP_CONFIG.PY
==============================================================================
This file defines the hyperparameter search space for the sweep.

Each dictionary in the `sweep_config` list represents one complete training
and evaluation run. The `run_sweep.py` script will iterate through this list.
"""

sweep_config = [
    # --- Run 1: Baseline ---
    {
        "run_name": "run_1_baseline",
        "learning_rate": 2e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 2: Higher Learning Rate ---
    {
        "run_name": "run_2_higher_lr",
        "learning_rate": 5e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 3: Deeper LoRA ---
    {
        "run_name": "run_3_deeper_lora",
        "learning_rate": 2e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 4: More Epochs ---
    {
        "run_name": "run_4_more_epochs",
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 5: Higher Dropout ---
    {
        "run_name": "run_5_higher_dropout",
        "learning_rate": 2e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "weight_decay": 0.01,
    },
]