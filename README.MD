## Aura-Mind: Maize Expert - Project Workflow

This document outlines the complete, end-to-end pipeline for training and preparing the "Aura-Mind: Maize Expert" model, a specialized version of Google's Gemma 3N fine-tuned to identify maize plant conditions.

The entire process is conducted using free cloud computing resources (Kaggle/Google Colab) and the Unsloth library for optimized training.

### Part 1: Data Preparation (Local Machine)

This phase involves collecting and organizing the image data on your local computer.

**Step 1: Data Collection & Curation**
Collect images for the target classes. For the Maize MVP, these are:
*   `maize_healthy`: Images of healthy maize plants, cobs, and leaves in various real-world conditions.
*   `phosphorus_deficiency`: Images showing the specific symptoms of phosphorus deficiency in maize.

**Step 2: Directory Structure**
Organize the curated images into a clear folder structure. This structure is critical for the automated processing steps later.
```
datasets/
└── train/
    ├── maize_healthy/
    │   ├── image_01.jpg
    │   ├── image_02.jpg
    │   └── ...
    └── phosphorus_deficiency/
        ├── image_A.jpg
        ├── image_B.jpg
        └── ...
```

**Step 3: Package the Dataset**
The entire `datasets/train` directory is compressed into a single `.zip` file for easy uploading to the cloud environment. A helper script is used for this.

*   **Command:**
    ```bash
    python create_zip.py
    ```
*   **Output:** A file named `maize_dataset.zip` (or similar).

### Part 2: Model Fine-Tuning (Kaggle Notebook)

This phase uses a Kaggle Notebook with a GPU accelerator (T4 x1) to fine-tune the base model.

**Step 1: Setup and Data Upload**
1.  Create a new Kaggle Notebook with a T4 GPU.
2.  Install all required libraries (`unsloth`, `triton`, `transformers`, etc.).
3.  Upload the `maize_dataset.zip` file using the `+ Add Data` UI.

**Step 2: Load and Prepare Data**
The notebook unzips the dataset and prepares it in the specific format required by the Unsloth library. Each image is paired with a conversational prompt in a Python list of dictionaries.

**Step 3: Load Base Model and Add Adapters**
The `unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit` model is loaded in 4-bit precision. PEFT LoRA adapters are then applied to the model's vision and language layers, preparing it for fine-tuning.

**Step 4: Train the Model**
The model is trained using the `trl.SFTTrainer` configured with Unsloth's specialized `UnslothVisionDataCollator`. Key training parameters:
*   **`max_steps`**: 400+ (to ensure sufficient learning)
*   **`learning_rate`**: 1e-4 - 5e-5 (a stable rate for vision tasks)
*   **`gradient_checkpointing`**: `False` (as required for Gemma 3N vision tuning)

The training process is monitored by observing the decrease in the training loss.

**Step 5: Merge and Push to Hub**
After training is complete, the trained LoRA adapters are merged into the base model. This final, complete model is then pushed to a public Hugging Face Hub repository for permanent storage and easy access. This is a critical step that solves local loading issues and provides a stable backup.

### Part 3: Model Conversion (New Kaggle/Colab Notebook)

This phase takes the fine-tuned model from the Hub and converts it into the final format required for the Android application.

**Step 1: Setup Conversion Environment**
A new notebook is created, and a different set of libraries are installed, including `tensorflow`, `onnx`, `onnx-tf`, and `tflite-support`.

**Step 2: Load Fine-Tuned Model from Hub**
The merged, fine-tuned model is loaded directly from its Hugging Face repository ID (e.g., `YourUsername/AgroSage-Maize-Expert-v1`). This is a clean and reliable way to access the model.

**Step 3: PyTorch -> ONNX Conversion**
The model is exported from its native PyTorch format into the ONNX (Open Neural Network Exchange) format. This acts as a universal intermediate representation.

**Step 4: ONNX -> TensorFlow -> TFLite Conversion**
The `.onnx` file is converted first into a TensorFlow SavedModel, and then into a quantized TensorFlow Lite (`.tflite`) model. The quantization step (e.g., to INT8) is crucial for reducing the model's size and making it run efficiently on mobile hardware.

**Step 5: Package to `.task` Format**
The final `.tflite` model is packaged with its associated metadata (like the class labels file, `labels.txt`) into a `.task` file using Google AI Edge tools. This `.task` file is the final deliverable, ready to be integrated into the Android application.