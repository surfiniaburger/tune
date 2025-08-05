import transformers
import inspect

# The base class for all Hugging Face models
from transformers import PreTrainedModel

print(f"--- Checking transformers version: {transformers.__version__} ---")

# A list to hold the names of all found model classes
model_classes = []

# A flag to check if the specific model is found
gemma3n_found = False

print("\n--- Searching for all available model classes in the library... ---")

# Iterate over all members of the transformers library
for name, obj in inspect.getmembers(transformers):
    # Check if the member is a class and if it's a subclass of PreTrainedModel
    # This is the most reliable way to identify a model class.
    # We also exclude the base class itself and any private classes.
    try:
        if inspect.isclass(obj) and issubclass(obj, PreTrainedModel) and name != "PreTrainedModel" and not name.startswith("_"):
            model_classes.append(name)
            if "Gemma3n" in name:
                gemma3n_found = True
    except (TypeError, ImportError):
        # Some objects might not be importable, we can safely ignore them
        continue

# Sort the list alphabetically for easy reading
model_classes.sort()

# Print the findings
print(f"\nFound {len(model_classes)} model classes.")

if gemma3n_found:
    print("\n✅ GREAT NEWS: Gemma3n model classes were found!")
else:
    print("\n❌ ATTENTION: Gemma3n model classes were NOT found. This indicates a problem with the transformers installation.")

print("\n--- List of all found model classes: ---")
for class_name in model_classes:
    print(f"- {class_name}")

print("\n--- Check complete ---")