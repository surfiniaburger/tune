"""
==============================================================================
SWEEP_CONFIG.PY (v4 - Focused on alpha=rank)
==============================================================================
This file defines a hyperparameter search space based on the core finding
that `lora_alpha` should be equal to `lora_rank`.

The sweep is structured into two main groups, one for each promising learning
rate (1e-4 and 3e-5). Within each group, it systematically varies other
key hyperparameters like LoRA rank, batch size, and training duration
to find the optimal configuration around this core principle.
"""

sweep_config = [
    # ==========================================================================
    # Group 1: More Aggressive Learning Rate (1e-4)
    # This group tests if faster learning combined with different model
    # capacities and batch sizes can yield the best results quickly.
    # ==========================================================================

    # --- Run 1: Baseline for LR=1e-4 ---
    # A solid middle-ground configuration for this learning rate.
    {
        "run_name": "run_01_lr1e-4_baseline_r16",
        "learning_rate": 1e-4,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 16,
        "lora_alpha": 16,  # alpha = rank
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 2: Lower Capacity (Low Rank) ---
    # Tests if a smaller, more regularized model performs better.
    {
        "run_name": "run_02_lr1e-4_low_rank_r8",
        "learning_rate": 1e-4,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 8,
        "lora_alpha": 8,  # alpha = rank
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 3: Higher Capacity (High Rank) ---
    # Tests if the model benefits from more trainable parameters.
    {
        "run_name": "run_03_lr1e-4_high_rank_r32",
        "learning_rate": 1e-4,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 32,
        "lora_alpha": 32,  # alpha = rank
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 4: Smaller Batch Size ---
    # More frequent gradient updates can sometimes help escape local minima.
    {
        "run_name": "run_04_lr1e-4_small_batch",
        "learning_rate": 1e-4,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,
        "lora_r": 16,
        "lora_alpha": 16,  # alpha = rank
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 5: More Training Epochs ---
    # See if longer training is beneficial with this higher learning rate.
    {
        "run_name": "run_05_lr1e-4_more_epochs",
        "learning_rate": 1e-4,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "lora_r": 16,
        "lora_alpha": 16,  # alpha = rank
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 6: High Capacity with More Regularization ---
    # A stress test: can a high-capacity model be controlled with more dropout?
    {
        "run_name": "run_06_lr1e-4_high_rank_high_dropout",
        "learning_rate": 1e-4,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 32,
        "lora_alpha": 32,  # alpha = rank
        "lora_dropout": 0.1, # Increased dropout
        "weight_decay": 0.01,
    },

    # ==========================================================================
    # Group 2: More Conservative Learning Rate (3e-5)
    # This group tests if a slower, more stable learning process leads to
    # a better final result, which is common for small datasets.
    # ==========================================================================

    # --- Run 7: Baseline for LR=3e-5 ---
    # The corresponding baseline for the more conservative learning rate.
    {
        "run_name": "run_07_lr3e-5_baseline_r16",
        "learning_rate": 3e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 16,
        "lora_alpha": 16,  # alpha = rank
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 8: Lower Capacity (Low Rank) ---
    {
        "run_name": "run_08_lr3e-5_low_rank_r8",
        "learning_rate": 3e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 8,
        "lora_alpha": 8,  # alpha = rank
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 9: Higher Capacity (High Rank) ---
    {
        "run_name": "run_09_lr3e-5_high_rank_r32",
        "learning_rate": 3e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "lora_r": 32,
        "lora_alpha": 32,  # alpha = rank
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 10: Larger Batch Size ---
    # Less frequent updates might lead to more stable training with a lower LR.
    {
        "run_name": "run_10_lr3e-5_large_batch",
        "learning_rate": 3e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 4,
        "lora_r": 16,
        "lora_alpha": 16,  # alpha = rank
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 11: More Training Epochs ---
    # Slower learning often requires more time to converge.
    {
        "run_name": "run_11_lr3e-5_more_epochs",
        "learning_rate": 3e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "lora_r": 16,
        "lora_alpha": 16,  # alpha = rank
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
    # --- Run 12: Low Capacity with More Epochs ---
    # A combination: does a smaller model catch up if given more time to train?
    {
        "run_name": "run_12_lr3e-5_low_rank_more_epochs",
        "learning_rate": 3e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "lora_r": 8,
        "lora_alpha": 8,  # alpha = rank
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
    },
]