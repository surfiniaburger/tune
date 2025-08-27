import os
import time
import logging
from gradio_client import Client, file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def diagnose_plant_from_huggingface(image_path_on_server: str) -> str | None:
    """
    Connects to the Hugging Face Gradio server, sends an image,
    and gets back a diagnosis. Includes a retry mechanism for cold starts.

    Args:
        image_path_on_server: The local path to the image file to be sent for diagnosis.

    Returns:
        The diagnosis text from the server, or None if an error occurs.
    """
    # --- Configuration for the retry mechanism ---
    MAX_RETRIES = 3
    # Delay between retries in seconds. Hugging Face spaces can take a minute to wake up.
    RETRY_DELAY_SECONDS = 45
    HF_SPACE_URL = "https://surfiniaburger-aura-mind-glow.hf.space/"

    logging.info(f"Connecting to Hugging Face Space: {HF_SPACE_URL}")
    try:
        # 1. Connect to your public Hugging Face Space
        client = Client(HF_SPACE_URL)
        logging.info("Connection to Hugging Face Space successful.")
    except Exception as e:
        logging.error(f"Fatal: Could not connect to the Gradio client. {e}", exc_info=True)
        return "Error: Could not connect to the diagnosis service."

    # 2. Loop for retry attempts
    for attempt in range(MAX_RETRIES):
        try:
            logging.info(f"--- Attempt {attempt + 1} of {MAX_RETRIES} ---")
            logging.info(f"Sending {image_path_on_server} for diagnosis...")

            # 3. Call the specific function (tool) on the server.
            # The client.predict method signature depends on the Gradio app's API.
            # We are calling the endpoint named "/get_diagnosis_and_remedy".
            result = client.predict(
                uploaded_image=file(image_path_on_server),
                feedback="Automated diagnosis from ADK Agent", # Example feedback
                api_name="/get_diagnosis_and_remedy"
            )

            logging.info("✅ Diagnosis received successfully!")
            logging.info(f"Result from server: {result}")

            # The result might be a complex object. We need to extract the text.
            # Based on Gradio's behavior, the result is often a tuple or a dictionary.
            # Let's assume the diagnosis text is the primary, first element.
            if isinstance(result, (list, tuple)) and len(result) > 0:
                diagnosis_text = result[0]
            elif isinstance(result, str):
                diagnosis_text = result
            else:
                 diagnosis_text = str(result)

            return diagnosis_text

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed. Error: {e}", exc_info=True)
            if attempt < MAX_RETRIES - 1:
                logging.warning(f"Server may be experiencing a cold start. Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logging.error("All retry attempts have failed. The server might be unavailable or has an error.")
                return "Error: The diagnosis service is currently unavailable after multiple retries."

if __name__ == '__main__':
    # This is a placeholder for a real image file path
    # In your app, this path would come from a user upload
    example_image_path = "sample_images/healthy_maize_test_1.jpg"

    # Create a dummy image file for testing if it doesn't exist
    if not os.path.exists(example_image_path):
        from PIL import Image
        print("Creating a dummy image for demonstration.")
        dummy_image_dir = os.path.dirname(example_image_path)
        if not os.path.exists(dummy_image_dir):
            os.makedirs(dummy_image_dir)
        dummy_image = Image.new('RGB', (100, 100), color = 'green')
        dummy_image.save(example_image_path)

    diagnosis = diagnose_plant_from_huggingface(example_image_path)
    if diagnosis:
        print("\n--- Diagnosis ---")
        print(diagnosis)
    else:
        print("\nFailed to get a diagnosis.")
