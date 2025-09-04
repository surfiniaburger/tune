"""
==============================================================================
SWEEP_CONFIG.PY (v2 - Expanded)
==============================================================================
This file defines the hyperparameter search space for the sweep.

This expanded configuration is designed to systematically explore key
hyperparameters for fine-tuning on a small dataset. It tests various
combinations of learning rates, LoRA configurations, batch sizes, epochs,
and regularization to find a balance between learning and overfitting.

Each dictionary represents one complete training run. The `run_sweep.py`
script will iterate through this list.
"""

sweep_config = [
    # --- Run 1: Solid Baseline ---
    # A conservative starting point with common parameters.
    {
        "run_name": "run_01_baseline",
        "learning_rate": 2e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 2: Lower Learning Rate ---
    # Slower learning can be beneficial for small datasets to avoid instability.
    {
        "run_name": "run_02_lower_lr",
        "learning_rate": 1e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 3: Higher Learning Rate ---
    # A more aggressive learning rate to see if the model can converge faster.
    {
        "run_name": "run_03_higher_lr",
        "learning_rate": 5e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 4: Lower Capacity LoRA ---
    # A smaller rank (r) reduces trainable parameters, which can prevent overfitting.
    {
        "run_name": "run_04_low_rank_lora",
        "learning_rate": 2e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 5: Higher Capacity LoRA ---
    # A larger rank allows the model to learn more complex patterns.
    {
        "run_name": "run_05_high_rank_lora",
        "learning_rate": 2e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 6: Smallest Batch Size ---
    # Varies the per-device batch size, which your script will compensate for
    # with more gradient accumulation steps to maintain the global batch size.
    {
        "run_name": "run_06_batch_size_1",
        "learning_rate": 2e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 7: Larger Batch Size ---
    # A larger per-device batch size reduces gradient accumulation steps.
    {
        "run_name": "run_07_batch_size_4",
        "learning_rate": 2e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 4,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 8: More Epochs ---
    # Training for longer might be necessary if the model learns slowly.
    {
        "run_name": "run_08_more_epochs",
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 9: Fewer Epochs ---
    # Training for just one epoch is a fast way to check for immediate learning and avoid overfitting.
    {
        "run_name": "run_09_fewer_epochs",
        "learning_rate": 2e-5,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 10: Increased Regularization (Dropout) ---
    # Higher dropout can help prevent the LoRA adapters from memorizing the training data.
    {
        "run_name": "run_10_high_dropout",
        "learning_rate": 2e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "weight_decay": 0.01,
    },
    # --- Run 11: Increased Regularization (Weight Decay) ---
    # Higher weight decay penalizes large weights to improve generalization.
    {
        "run_name": "run_11_high_weight_decay",
        "learning_rate": 2e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "weight_decay": 0.05,
    },
    # --- Run 12: High Capacity LoRA + More Regularization ---
    # A combination to test if a more powerful model can be tamed with stronger regularization.
    {
        "run_name": "run_12_high_rank_and_dropout",
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "weight_decay": 0.01,
    },
]