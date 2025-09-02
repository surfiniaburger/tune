"""
==============================================================================
SWEEP_CONFIG.PY (v3 - STABLE LoRA CONFIG)
==============================================================================
"""

from itertools import product

# Define search space
param_grid = {
    "learning_rate": [1e-5, 2e-5],
    "num_train_epochs": [2],
    "per_device_train_batch_size": [2],
    "lora_r": [8, 16, 32],
    # lora_alpha is now determined by lora_r
    "lora_dropout": [0.05, 0.1],
    "weight_decay": [0.01],
}

# Generate full sweep
sweep_config = []
# Get all possible combinations of the grid parameters
param_combinations = list(product(*param_grid.values()))

# Manually create the config for each run
for combo in param_combinations:
    # Match combo values back to their keys
    config = dict(zip(param_grid.keys(), combo))
    
    # --- FIX: Set lora_alpha equal to lora_r for stability ---
    config["lora_alpha"] = config["lora_r"] 
    
    run_name = (
        f"lr{config['learning_rate']}_"
        f"r{config['lora_r']}_a{config['lora_alpha']}_"
        f"do{config['lora_dropout']}_"
        f"wd{config['weight_decay']}_"
        f"ep{config['num_train_epochs']}"
    )
    config["run_name"] = run_name
    sweep_config.append(config)