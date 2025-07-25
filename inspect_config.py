from transformers import AutoConfig

# Path to your local model
pytorch_model_path = "./finetuned_model_for_conversion"

print(f"--- Inspecting config for model at: {pytorch_model_path} ---")

try:
    # Load the configuration exactly as the main script does
    config = AutoConfig.from_pretrained(pytorch_model_path, trust_remote_code=True)

    # Print all the attributes of the loaded config object
    print("\n[SUCCESS] Config loaded. Attributes are:")
    for key, value in config.__dict__.items():
        print(f"  - {key}: {value}")

except Exception as e:
    print(f"\n[ERROR] Failed to load or inspect config: {e}")

print("\n--- Inspection complete ---")