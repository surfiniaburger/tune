# You would have this code in your Cloud Run application
# Make sure to have 'gradio_client' in your requirements.txt
from gradio_client import Client, file
import os
import time # Import the time module for delays

def diagnose_plant_from_cloud_run(image_path_on_server):
    """
    Connects to the Hugging Face MCP server, sends an image,
    and gets back a diagnosis. Includes a retry mechanism for cold starts.
    """
    # --- Configuration for the retry mechanism ---
    MAX_RETRIES = 3
    # Delay between retries in seconds. Hugging Face spaces can take a minute to wake up.
    RETRY_DELAY_SECONDS = 45

    print("Connecting to Hugging Face Space...")
    try:
        # 1. Connect to your public Hugging Face Space
        client = Client("https://surfiniaburger-aura-mind-glow.hf.space/")
        print("Connection successful.")
    except Exception as e:
        print(f"Fatal: Could not connect to the Gradio client. {e}")
        return None

    # 2. Loop for retry attempts
    for attempt in range(MAX_RETRIES):
        try:
            print(f"--- Attempt {attempt + 1} of {MAX_RETRIES} ---")
            print(f"Sending {image_path_on_server} for diagnosis...")
            
            # 3. Call the specific function (tool) on the server.
            result = client.predict(
                uploaded_image=file(image_path_on_server),
                feedback="Automated diagnosis from Cloud Run",
                api_name="/get_diagnosis_and_remedy"
            )

            print("✅ Diagnosis received successfully!")
            print(result)
            return result  # If successful, return the result and exit the function

        except Exception as e:
            print(f"Attempt {attempt + 1} failed. Error: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Server may be experiencing a cold start. Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print("All retry attempts have failed. The server might be unavailable or has an error.")
                return None # All retries failed, exit the function

# --- Example Usage (within your Cloud Run app) ---
if __name__ == '__main__':
    # This is a placeholder for a real image file path
    # In your app, this path would come from a user upload
    example_image_path = "path/to/your/image.png"

    # Create a dummy image file for testing if it doesn't exist
    if not os.path.exists(example_image_path):
        from PIL import Image
        print("Creating a dummy image for demonstration.")
        if not os.path.exists("path/to/your"):
            os.makedirs("path/to/your")
        dummy_image = Image.new('RGB', (100, 100), color = 'red')
        dummy_image.save(example_image_path)

    diagnose_plant_from_cloud_run(example_image_path)