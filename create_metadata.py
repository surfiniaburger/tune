import os
import csv

def generate_metadata_csv(main_directory):
    """
    Generates a metadata.csv file for an image dataset structured in subdirectories,
    where each subdirectory name is the label for the images within it.

    Args:
        main_directory (str): The path to the main directory containing subdirectories of images.
    """
    
    # Check if the main directory exists
    if not os.path.isdir(main_directory):
        print(f"Error: The directory '{main_directory}' does not exist.")
        return

    metadata = []
    header = ['file_name', 'text']

    # Get the list of subdirectories (our labels)
    subdirectories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]

    if not subdirectories:
        print(f"Error: No subdirectories found in '{main_directory}'.")
        return

    # Loop through each subdirectory
    for label in subdirectories:
        class_dir = os.path.join(main_directory, label)
        
        # Loop through each file in the subdirectory
        for filename in os.listdir(class_dir):
            # Check if the file is a common image type
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # Create the relative path, e.g., "maize_healthy/image_01.jpg".
                # Using forward slashes for better compatibility (especially with web platforms).
                relative_path = f"{label}/{filename}"
                metadata.append([relative_path, label])

    # The output file will be created inside the main directory
    output_csv_path = os.path.join(main_directory, 'metadata.csv')

    # Write the data to the CSV file
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(metadata)
        print(f"Successfully generated metadata.csv with {len(metadata)} entries at: {output_csv_path}")
    except IOError as e:
        print(f"Error writing to file: {e}")


# --- HOW TO USE THE SCRIPT ---

# 1. Set this variable to the path of your main dataset folder.
dataset_path = 'maize_dataset' 

# 2. Run the function.
generate_metadata_csv(dataset_path)