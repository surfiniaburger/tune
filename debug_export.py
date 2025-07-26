import torch
import traceback
import os
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig
from optimum.exporters.onnx import onnx_export_from_model
from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.config import TextDecoderOnnxConfig
from optimum.utils import (
    DummyPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyVisionInputGenerator,
)
from typing import Dict, Optional
from optimum.exporters.onnx.model_patcher import ModelPatcher
import types

# --- Configuration ---
pytorch_model_path = "./finetuned_model_for_conversion"
onnx_output_path = "./onnx_model"
print(f"PyTorch model path: {pytorch_model_path}")
print(f"ONNX output path: {onnx_output_path}")

# ==============================================================================
# CELL 0: ONNX EXPORT PATCHER FOR GEMMA-3N
# ==============================================================================
def patched_project_per_layer_inputs(
    self,  # self is Gemma3nTextModel instance
    inputs_embeds: torch.Tensor,
    per_layer_inputs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # This is the original implementation from transformers/models/gemma3n/modeling_gemma3n.py
    per_layer_projection: torch.Tensor = self.per_layer_model_projection(inputs_embeds)
    per_layer_projection *= self.per_layer_projection_scale.to(
        dtype=inputs_embeds.dtype, device=per_layer_projection.device
    )
    per_layer_projection = per_layer_projection.reshape(
        *inputs_embeds.shape[:-1],
        self.config.num_hidden_layers,
        self.hidden_size_per_layer_input,
    )
    per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

    if per_layer_inputs is None:
        return per_layer_projection

    # ONNX PATCH: The original code has a data-dependent `if` statement here:
    # `if per_layer_projection.shape != per_layer_inputs.shape:`
    # This is not traceable. We remove the `if` and always perform the slice,
    # which is safe for tracing as we control the dummy inputs.
    per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]

    return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale.to(
        dtype=inputs_embeds.dtype, device=per_layer_projection.device
    )


class CustomGemma3NPatcher(ModelPatcher):
    def __enter__(self):
        super().__enter__()
        self.original_project_per_layer_inputs = self._model.model.language_model.project_per_layer_inputs
        self._model.model.language_model.project_per_layer_inputs = types.MethodType(patched_project_per_layer_inputs, self._model.model.language_model)

    def __exit__(self, exc_type, exc_value, traceback):
        self._model.model.language_model.project_per_layer_inputs = self.original_project_per_layer_inputs
        super().__exit__(exc_type, exc_value, traceback)


# ==============================================================================
# CELL 1: THE DEFINITIVE DUMMY INPUT GENERATOR
# ==============================================================================
class CustomDummyTextInputGenerator(DummyTextInputGenerator):
    """
    A robust dummy text input generator that MANUALLY creates all required text tensors
    using hardcoded, standard shapes to bypass all library state issues.
    """
    # The signature is updated to accept `input_name` and return a single Tensor.
    def generate(self, input_name: str, *args, **kwargs) -> torch.Tensor:
        """
        This method constructs the dummy text inputs manually with fixed shapes.
        It now correctly returns only the tensor requested by `input_name`.
        """
        print(f"\n--- [DEBUG] Inside CustomDummyTextInputGenerator.generate (requesting '{input_name}') ---")

        # We still generate all related inputs here, but will only return the one requested.
        # We hardcode standard dummy shapes to completely avoid library state issues.
        batch_size = 2
        sequence_length = 16  # A standard default for dummy inputs
        vocab_size = self.normalized_config.vocab_size
        print(f"[DEBUG] Generating tensors with HARDCODED shapes: batch_size={batch_size}, sequence_length={sequence_length}")

        # Manually create the `input_ids` and `attention_mask` tensors.
        manually_created_input_ids = torch.randint(low=3, high=vocab_size, size=(batch_size, sequence_length), dtype=torch.long)
        manually_created_attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
        
        # Set the first token to BOS (Beginning of Sequence) ID: 2
        manually_created_input_ids[:, 0] = 2 # BOS token

        # Inject the special image token by reconstructing the `input_ids` tensor.
        image_token_id = 262145 # From config inspection (inspect_library.py)
        # The number of image tokens to insert is now read from the config.
        num_image_tokens = self.normalized_config.vision_soft_tokens_per_image
        part1 = manually_created_input_ids[:, :1]
        # Adjust the slice to make room for all the image tokens.
        part2 = manually_created_input_ids[:, 1 + num_image_tokens:]
        image_token_tensor = torch.full((batch_size, num_image_tokens), image_token_id, dtype=torch.long)
        new_input_ids = torch.cat([part1, image_token_tensor, part2], dim=1)
        print(f"[DEBUG] Reconstructed new_input_ids shape: {new_input_ids.shape}")
        
        # Store all generated text-related tensors in a dictionary
        all_text_inputs = {
            "input_ids": new_input_ids,
            "attention_mask": manually_created_attention_mask,
        }

        # Return only the tensor that was asked for.
        if input_name not in all_text_inputs:
            raise ValueError(f"{self.__class__.__name__} was asked to generate a dummy input for '{input_name}' but it only supports {list(all_text_inputs.keys())}.")

        print(f"--- [DEBUG] Leaving CustomDummyTextInputGenerator.generate (returning tensor for '{input_name}') ---\n")
        return all_text_inputs[input_name]

# ==============================================================================
# CELL 2 & 3 (Unchanged but confirmed correct)
# ==============================================================================
print("\nStep 1: Defining a custom ONNX configuration...")
class CustomGemma3NMultimodalOnnxConfig(TextDecoderOnnxConfig):
    _MODEL_PATCHER = CustomGemma3NPatcher
    DUMMY_INPUT_GENERATOR_CLASSES = (CustomDummyTextInputGenerator, DummyVisionInputGenerator, DummyPastKeyValuesGenerator)
    from optimum.utils import NormalizedTextConfig
    # Add `vision_soft_tokens_per_image` to the normalized config so our generator can access it.
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_layers="num_hidden_layers",
        num_attention_heads="num_attention_heads",
        hidden_size="hidden_size",
        vocab_size="vocab_size",
        vision_soft_tokens_per_image="vision_soft_tokens_per_image",
        allow_new=True,
    )
    def __init__(self, config: PretrainedConfig, task: str = "default", **kwargs):
        # Inject the number of vision tokens from the main config's vision_config
        # into the text_config before passing it to the parent constructor.
        # This makes it available to the NormalizedTextConfig.
        config.text_config.vision_soft_tokens_per_image = config.vision_soft_tokens_per_image
        super().__init__(config=config.text_config, task=task, **kwargs)

    def get_inputs_and_outputs(self, use_past: bool) -> Dict[str, Dict[int, str]]:
        text_inputs, vision_input = {"input_ids": {0:"batch_size", 1:"sequence_length"}, "attention_mask": {0:"batch_size", 1:"sequence_length"}}, {"pixel_values": {0:"batch_size", 1:"num_channels", 2:"height", 3:"width"}}
        past_key_values = {}
        if use_past:
            text_inputs["input_ids"], text_inputs["attention_mask"] = {0:"batch_size", 1:"1"}, {0:"batch_size", 1:"past_sequence_length + 1"}
            for i in range(self._normalized_config.num_layers):
                past_key_values[f"past_key_values.{i}.key"], past_key_values[f"past_key_values.{i}.value"] = {0:"batch_size", 2:"past_sequence_length"}, {0:"batch_size", 2:"past_sequence_length"}
        return {**text_inputs, **vision_input, **past_key_values}
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]: return self.get_inputs_and_outputs(self.use_past)
    @property
    def torch_to_onnx_output_map(self): return {"logits":"logits", "past_key_values":"present"} if self.use_past else {"logits":"logits"}

def get_submodels(model: AutoModelForCausalLM) -> Dict[str, AutoModelForCausalLM]:
    return {"decoder_model": model, "decoder_with_past_model": model}

# ==============================================================================
# CELL 4: PREPARE AND RUN THE ONNX EXPORT
# ==============================================================================
print("\nStep 2: Preparing and running the ONNX export...")
try:
    # --- Pre-load check and fix for empty index files ---
    # The JSONDecodeError occurs if an empty `model.safetensors.index.json` exists,
    # as transformers tries to parse it for sharded checkpoints.
    index_path = Path(pytorch_model_path) / "model.safetensors.index.json"
    if index_path.exists() and index_path.stat().st_size == 0:
        print(f"⚠️  Found an empty index file at: {index_path}")
        print("   This can cause loading errors. Removing it to proceed.")
        os.remove(index_path)
        print("   ✅ Empty index file removed.")

    # Leverage GPU if available to reduce CPU memory pressure
    # --- MPS DEBUGGING ---
    # The error "Placeholder storage has not been allocated on MPS device!"
    # indicates a problem with how tensors are being materialized on the Apple
    # Silicon GPU. As a robust workaround, we will force the export to run
    # entirely on the CPU. This avoids the MPS backend bug and is often stable
    # on Macs due to unified memory.
    device = "cpu"
    print(f"Using device: {device}")

    # Load the model and config ONCE to have better control over memory.
    print("Loading model and config from disk...")
    main_config = AutoConfig.from_pretrained(pytorch_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        pytorch_model_path,
        config=main_config,
        trust_remote_code=True,
    ).to(device)  # Move the model to the GPU immediately
    print("Model loaded.")

    custom_onnx_configs = {
        "decoder_model": CustomGemma3NMultimodalOnnxConfig(config=main_config, task="text-generation", use_past=False),
        "decoder_with_past_model": CustomGemma3NMultimodalOnnxConfig(config=main_config, task="text-generation", use_past=True),
    }
    # Use the more direct `onnx_export_from_model` which takes a pre-loaded model object.
    onnx_export_from_model(
         model=model,
         output=Path(onnx_output_path),
         task="text-generation-with-past",
         custom_onnx_configs=custom_onnx_configs,
         fn_get_submodels=get_submodels,
         opset=15,
         # Disable validation to prevent out-of-memory (OOM) errors during export.
         do_validation=False,
         device=device,  # Ensure the exporter uses the GPU
     )
    # main_export(
    #     model_name_or_path=pytorch_model_path, output=onnx_output_path, task="text-generation-with-past",
    #     trust_remote_code=True, custom_onnx_configs=custom_onnx_configs, fn_get_submodels=get_submodels, opset=14, do_validation=False,
    # )
    print("\n✅ ONNX conversion process completed successfully!")
    print(f"   The exported model is saved in: {Path(onnx_output_path).resolve()}")
except Exception:
    print(f"\n❌ An error occurred during the ONNX conversion process."); print("--- FULL TRACEBACK ---"); traceback.print_exc(); print("--- END OF TRACEBACK ---")