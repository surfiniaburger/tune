# Aura-Mind: Maize Expert - Project Workflow & Documentation

## Overview

This document outlines the complete, end-to-end pipeline for creating the **Aura-Mind: Maize Expert**, a specialized multimodal AI model for on-device agricultural diagnostics. The project leverages Google's Gemma 3N architecture, fine-tuned on a custom dataset of Nigerian maize plant conditions.

The workflow is divided into three distinct phases, moving from local data preparation to cloud-based fine-tuning, and finally to a local, controlled conversion for Android deployment.

---

### **Part 1: Data Curation & Preparation (Local Machine)**

The foundation of a high-quality model is high-quality data. This phase focuses on collecting, organizing, and packaging a focused, real-world dataset.

**Step 1: Data Collection (Maize MVP)**
For the initial Minimum Viable Product (MVP), the scope was strategically focused on a single, high-impact crop: Maize. Data was collected for two critical classes:
*   **`maize_healthy`**: A diverse set of images showing healthy maize plants, leaves, and cobs in various market and field conditions in Lagos, Nigeria.
*   **`phosphorus_deficiency`**: A set of academic images showing the clear visual symptoms of phosphorus deficiency in maize.

**Step 2: Directory Structure**
The curated images are organized into a clean, class-per-folder structure, which is essential for automated processing.
```
datasets/
└── train/
    ├── maize_healthy/
    │   ├── image_01.jpg
    │   └── ...
    └── phosphorus_deficiency/
        ├── image_A.jpg
        └── ...
```

**Step 3: Packaging for the Cloud**
To ensure data integrity and easy transfer, the entire `datasets/train` directory is compressed into a single archive.
*   **Script:** `create_zip.py`
*   **Output:** `maize_dataset.zip`

---

### **Part 2: Model Fine-Tuning (Kaggle Notebook)**

This phase uses a Kaggle Notebook with a single GPU accelerator (T4 x1) to leverage the powerful, memory-optimized Unsloth library for fine-tuning.

**Step 1: Environment Setup**
A Kaggle notebook is initialized, and all required libraries are installed, including `unsloth`, `peft`, `bitsandbytes`, and a specific version of `triton` to ensure compatibility. The `maize_dataset.zip` is uploaded and unzipped.

**Step 2: Data Loading & Formatting**
The unzipped images are loaded and formatted into the specific **Python list of dictionaries** that the Unsloth vision pipeline requires. Each entry contains a PIL Image object and a corresponding conversational prompt. This meticulous data preparation, discovered through extensive debugging, is critical for success.

**Step 3: Model Loading & LoRA Configuration**
The `unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit` (2B parameter) model is loaded in 4-bit precision. PEFT LoRA adapters are configured to fine-tune both the `vision_layers` and `language_layers`. A key finding was that `gradient_checkpointing` must be set to `False` for Gemma 3N vision tuning with Unsloth.

**Step 4: Training**
The model is trained using the `trl.SFTTrainer`, but it is configured with Unsloth's specialized **`UnslothVisionDataCollator`**. This custom component handles all on-the-fly data processing and is essential for a successful run. The training loss is monitored, which plummets to near-zero, indicating the model has successfully learned the training data.

**Step 5: Merge and Push to Hugging Face Hub**
Upon completion, the trained LoRA adapters are merged into the base model. This final, complete "Maize Expert" model is then pushed to a public Hugging Face Hub repository (`surfiniaburger/AuraMind-Maize-Expert-v1`). This provides a permanent, version-controlled backup and simplifies the next phase by providing a stable, universal access point.

---

### **Part 3: Model Conversion to `.task` Format (Local Machine)**

This final, highly technical phase converts the trained PyTorch model into a format compatible with the Google AI Edge toolkit for on-device deployment. This is performed on a local Mac to ensure full control over the complex dependency chain, bypassing unresolved conflicts in cloud environments.

**Step 1: Create a Dedicated Conversion Environment**
A new, clean Python virtual environment (`.venv-converter`) is created locally. A precise sequence of installations is performed to ensure library compatibility, including specific versions of `tensorflow`, `protobuf`, and the `ai-edge-torch` library cloned from its repository.

**Step 2: Download the Fine-Tuned Model**
The "Maize Expert" model is downloaded from the Hugging Face Hub into a local directory (`finetuned_model_for_conversion`).

**Step 3: The "Two-Stage" TFLite Conversion**
Due to the model's multimodal architecture, a direct conversion is not possible. A custom script, `convert_my_model.py`, was developed to perform a two-stage conversion:
1.  **Vision Encoder:** The `vision_tower` is isolated from the full model. A dummy image input is created, and the `ai_edge_torch.convert` function is used to produce a quantized `vision_encoder_dynamic_int8.tflite` file.
2.  **Text Decoder:** The language-only weights are loaded into a clean, text-only model structure. The `ai_edge_torch.generative.utilities.converter` is then used to convert this language model into a separate `text_decoder_dynamic_int8.tflite` file.

**Step 4: Tokenizer Preparation**
A separate script from the `ai-edge-torch` library (`tokenizer_to_sentencepiece.py`) is used to extract and format the tokenizer from the Hugging Face repo into the required `tokenizer.model` file.

**Step 5: Bundling the Final `.task` File**
The final step uses the `mediapipe` library's bundler. A custom Python script, `bundle_task_file.py`, combines the generated TFLite models and the `tokenizer.model` into a single, deployable `agro_sage_maize_expert.task` file. This configuration includes critical metadata, such as the `prompt_prefix`, `start_token`, and `stop_tokens`, ensuring the model behaves correctly within the Android app.