# data_utils.py
import random
from PIL import Image, ImageEnhance, ImageOps

# The "Translator" Dictionary mapping folder names to clean, human-readable names.
# This is the single source of truth for our class labels.
CLASS_NAME_MAPPING = {
    # Diseases from the Abuja Dataset
    "anthracnose": "Sorghum Anthracnose",
    "cercospora_leaf_spot": "Cowpea Cercospora Leaf Spot",
    "phosphorus_deficiency": "Maize Phosphorus Deficiency",
    "rice_brown_leaf_spot": "Rice Brown Leaf Spot",
    "sunflower_leaf_blight": "Sunflower Leaf Blight",

    # Your new classes from market data
    "pepper_late_blight": "Pepper Late Blight",
    "tomato_late_blight": "Tomato Late Blight", # You will create this folder

    # Your new healthy classes
    "peppers_healthy": "Healthy Pepper Plant",
    "tomatoes_healthy": "Healthy Tomato Plant", # You will create this folder
    
    # Placeholder for healthy Abuja crops (add images to these folders)
    "sorghum_healthy": "Healthy Sorghum Plant",
    "cowpea_healthy": "Healthy Cowpea Plant",
    "maize_healthy": "Healthy Maize Plant",
    "rice_healthy": "Healthy Rice Plant",
    "sunflower_healthy": "Healthy Sunflower Plant",
}

def augment_image(image: Image.Image) -> Image.Image:
    """
    Applies a series of random augmentations to a Pillow image.
    This simulates real-world variations.
    """
    # 1. Random Rotation
    angle = random.uniform(-15, 15)
    image = image.rotate(angle)

    # 2. Random Horizontal Flip (50% chance)
    if random.random() > 0.5:
        image = ImageOps.mirror(image)

    # 3. Random Brightness (adjusts between 70% and 130% of original)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))

    # 4. Random Contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))
    
    return image