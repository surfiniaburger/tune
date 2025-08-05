import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

# 1. Set the path to your local fine-tuned model directory.
# Based on your image, the model is in the 'finetuned_model_for_conversion' folder.
model_path = "./finetuned_model_for_conversion"

try:
    # 2. Load the model and processor from the specified path.
    model, processor = load(model_path)
    config = model.config

    # 3. Prepare your inputs.
    #    - Replace with the actual path to your image and audio files.
    #    - This model can handle both image and audio inputs.
    image_path = ["./training.png"]  # Replace with your image file
    audio_path = ["./speech.mp3"]  # Replace with your audio file

    # 4. Create a prompt.
    #    - Ask the model to describe what it perceives in the inputs.
    #    - You can leave the prompt empty "" if you want the model to freely describe the inputs.
    prompt = "Describe what you see and hear in the provided files."

    # 5. Apply the chat template.
    #    This formats the prompt correctly for the model, accounting for images and audio.
    formatted_prompt = apply_chat_template(
        processor, config, prompt,
        num_images=len(image_path),
        num_audios=len(audio_path)
    )

    # 6. Generate the output from the model.
    output = generate(
        model,
        processor,
        formatted_prompt,
        image=image_path,
        audio=audio_path,
        verbose=True # Set to True to see generation details
    )

    # 7. Print the final output.
    print("\n--- Model Output ---\n")
    print(output)

except FileNotFoundError:
    print(f"Error: Model not found at path '{model_path}'.")
    print("Please ensure the path is correct and points to the directory containing the model files.")
except Exception as e:
    print(f"An error occurred: {e}")