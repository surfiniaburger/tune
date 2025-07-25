import torch
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig
from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.config import TextDecoderOnnxConfig
from optimum.utils import (
    NormalizedConfig,
    NormalizedTextConfig,
    DummyPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyVisionInputGenerator,
)
from typing import Dict

# --- Configuration ---
# Path to the PyTorch model you downloaded
pytorch_model_path = "./finetuned_model_for_conversion"
# Path where the final ONNX model will be saved
onnx_output_path = "./onnx_model"

print(f"PyTorch model path: {pytorch_model_path}")
print(f"ONNX output path: {onnx_output_path}")

# ==============================================================================
# CELL 1: CREATE A CUSTOM DUMMY INPUT GENERATOR (FINAL, ROBUST VERSION)
# ==============================================================================
class CustomDummyTextInputGenerator(DummyTextInputGenerator):
    """
    A custom dummy text input generator that injects the known special image token
    into the `input_ids` by constructing a new tensor, which is safe for ONNX tracing.
    """
    def generate(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        # Let the parent class's generate method handle all arguments.
        inputs = super().generate(*args, **kwargs)
        
        # The model needs a placeholder for the image. From inspection, we know
        # the special image token ID is 262145.
        image_token_id = 262145
        
        # *** THE DEFINITIVE FIX: RECONSTRUCT THE TENSOR ***
        original_input_ids = inputs["input_ids"]
        batch_size = original_input_ids.shape[0]

        # 1. Slice the original tensor into parts.
        #    Part 1: The first token (usually BOS - Start of Sequence)
        part1 = original_input_ids[:, :1]
        #    Part 2: Everything after the insertion point
        part2 = original_input_ids[:, 2:]

        # 2. Create a new 2D tensor for our special image token.
        #    Shape must be [batch_size, 1] for concatenation.
        image_token_tensor = torch.full((batch_size, 1), image_token_id, dtype=torch.long)

        # 3. Concatenate the pieces along the sequence dimension (dim=1).
        new_input_ids = torch.cat([part1, image_token_tensor, part2], dim=1)

        # 4. Replace the old tensor with our newly constructed, correct tensor.
        inputs["input_ids"] = new_input_ids
        
        return inputs

# ==============================================================================
# CELL 2: DEFINE THE CUSTOM ONNX CONFIGURATION
# ==============================================================================
print("\nStep 1: Defining a custom ONNX configuration...")

class CustomGemma3NMultimodalOnnxConfig(TextDecoderOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        CustomDummyTextInputGenerator,
        DummyVisionInputGenerator,
        DummyPastKeyValuesGenerator,
    )
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_layers="num_hidden_layers",
        num_attention_heads="num_attention_heads",
        hidden_size="hidden_size",
        vocab_size="vocab_size",
    )
    def __init__(self, config: PretrainedConfig, task: str = "default", **kwargs):
        super().__init__(config=config.text_config, task=task, **kwargs)

    def get_inputs_and_outputs(self, use_past: bool) -> Dict[str, Dict[int, str]]:
        text_inputs = {"input_ids": {0: "batch_size", 1: "sequence_length"}, "attention_mask": {0: "batch_size", 1: "sequence_length"}}
        vision_input = {"pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}}
        past_key_values = {}
        if use_past:
            text_inputs["input_ids"] = {0: "batch_size", 1: "1"}
            text_inputs["attention_mask"] = {0: "batch_size", 1: "past_sequence_length + 1"}
            for i in range(self._normalized_config.num_layers):
                past_key_values[f"past_key_values.{i}.key"] = {0: "batch_size", 2: "past_sequence_length"}
                past_key_values[f"past_key_values.{i}.value"] = {0: "batch_size", 2: "past_sequence_length"}
        return {**text_inputs, **vision_input, **past_key_values}

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return self.get_inputs_and_outputs(use_past=self.use_past)

    @property
    def torch_to_onnx_output_map(self):
        return {"logits": "logits", "past_key_values": "present"} if self.use_past else {"logits": "logits"}

# ==============================================================================
# CELL 3: DEFINE THE SUBMODEL MAPPING FUNCTION
# ==============================================================================
def get_submodels(model: AutoModelForCausalLM) -> Dict[str, AutoModelForCausalLM]:
    return {"decoder_model": model, "decoder_with_past_model": model}

# ==============================================================================
# CELL 4: PREPARE AND RUN THE ONNX EXPORT
# ==============================================================================
print("\nStep 2: Preparing and running the ONNX export...")

try:
    main_config = AutoConfig.from_pretrained(pytorch_model_path, trust_remote_code=True)
    custom_onnx_configs = {
        "decoder_model": CustomGemma3NMultimodalOnnxConfig(config=main_config, task="text-generation", use_past=False),
        "decoder_with_past_model": CustomGemma3NMultimodalOnnxConfig(config=main_config, task="text-generation", use_past=True),
    }

    main_export(
        model_name_or_path=pytorch_model_path,
        output=onnx_output_path,
        task="text-generation-with-past",
        trust_remote_code=True,
        custom_onnx_configs=custom_onnx_configs,
        fn_get_submodels=get_submodels,
        opset=14,
    )
    print("\n✅ ONNX conversion process completed successfully!")
    print(f"   The exported model is saved in: {Path(onnx_output_path).resolve()}")

except Exception as e:
    print(f"\n❌ An error occurred during the ONNX conversion process: {e}")
    print("   Please check the following:")
    print("   1. Ensure all dependencies are installed correctly (`pip install \"optimum[exporters]\" transformers torch accelerate pillow timm`).")
    print("   2. Verify that the `pytorch_model_path` is correct.")
    print("   3. Your model might have a specific operator not supported by the default ONNX opset. You can try adjusting the `opset` parameter.")