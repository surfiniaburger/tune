"""
==============================================================================
SWEEP_CONFIG.PY (v2 - Recommended Sweep)
==============================================================================
This file defines the hyperparameter search space for the sweep.

Instead of fixed runs, we define a *grid* of parameter combinations that
`run_sweep.py` will iterate through systematically.
"""

from itertools import product

# Define search space
param_grid = {
    "learning_rate": [1e-5, 2e-5],
    "num_train_epochs": [2, 3],
    "per_device_train_batch_size": [2],  # keep small for Kaggle GPU
    "lora_r": [8, 16, 32],
    "lora_alpha": [32],
    "lora_dropout": [0.05, 0.1],
    "weight_decay": [0.01],
}

# Generate full sweep (Cartesian product of grid)
sweep_config = []
for values in product(*param_grid.values()):
    config = dict(zip(param_grid.keys(), values))
    run_name = (
        f"lr{config['learning_rate']}_"
        f"r{config['lora_r']}_a{config['lora_alpha']}_"
        f"do{config['lora_dropout']}_"
        f"wd{config['weight_decay']}_"
        f"ep{config['num_train_epochs']}"
    )
    config["run_name"] = run_name
    sweep_config.append(config)
