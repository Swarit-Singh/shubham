import cv2
import numpy as np
import logging
from utils import setup_logging # Use logging from utils

# Call setup_logging if this module is run standalone or imported early
# setup_logging() # Or let app.py handle main logging setup

def load_and_preprocess(image_path, target_size=None, color_mode='grayscale'):
    """
    Loads an image, converts color, and optionally resizes.
    Args:
        image_path (str): Path to the image file.
        target_size (tuple, optional): Desired (width, height). Defaults to None (no resize).
        color_mode (str): 'grayscale' or 'color'.
    Returns:
        np.ndarray: Preprocessed image as a NumPy array, or None on error.
    """
    try:
        if color_mode == 'grayscale':
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        elif color_mode == 'color':
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV loads BGR, convert to RGB
        else:
            logging.error(f"Unsupported color mode: {color_mode}")
            return None

        if img is None:
            logging.error(f"Failed to load image from {image_path}")
            return None

        if target_size:
            if not (isinstance(target_size, tuple) and len(target_size) == 2):
                logging.error("target_size must be a tuple of (width, height).")
                # Don't resize if target_size is invalid, proceed with original size
            else:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        logging.info(f"Image {image_path} loaded and preprocessed. Shape: {img.shape}")
        return img

    except Exception as e:
        logging.error(f"Error during preprocessing of {image_path}: {e}")
        return None

def save_image_cv(image_array, file_path):
    """
    Saves a NumPy array image using OpenCV.
    Args:
        image_array (np.ndarray): Image to save.
        file_path (str): Path to save the image.
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # OpenCV handles BGR for color saving by default.
        # If image_array is RGB, it might need conversion if saving with cv2.imwrite
        # For grayscale, it's fine.
        if image_array.ndim == 3 and image_array.shape[2] == 3: # Color image
            # If it was RGB from load_and_preprocess, convert back to BGR for OpenCV saving
            image_to_save = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else: # Grayscale
            image_to_save = image_array

        # Ensure uint8 for saving, common requirement for image formats
        if image_to_save.dtype != np.uint8:
            image_to_save = np.clip(image_to_save, 0, 255).astype(np.uint8)

        cv2.imwrite(file_path, image_to_save)
        logging.info(f"Image saved to {file_path} using OpenCV.")
        return True
    except Exception as e:
        logging.error(f"Error saving image with OpenCV to {file_path}: {e}")
        return False

# Example Usage
if __name__ == '__main__':
    setup_logging(level=logging.DEBUG) # More verbose for testing
    
    # Create a dummy image file for testing
    dummy_img_name = "dummy_test_image.png"
    dummy_array = np.random.randint(0, 256, (100, 150), dtype=np.uint8) # Grayscale
    cv2.imwrite(dummy_img_name, dummy_array)

    # Test loading grayscale
    gray_img = load_and_preprocess(dummy_img_name, target_size=(75, 50), color_mode='grayscale')
    if gray_img is not None:
        logging.debug(f"Loaded grayscale image shape: {gray_img.shape}")
        save_image_cv(gray_img, "dummy_grayscale_resized.png")

    # Test loading color (will be grayscale as source is grayscale, but tests path)
    # To test color properly, save a color dummy image first
    dummy_color_array = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
    cv2.imwrite("dummy_color_image.png", dummy_color_array) # Saved as BGR by OpenCV
    
    color_img = load_and_preprocess("dummy_color_image.png", color_mode='color')
    if color_img is not None:
        logging.debug(f"Loaded color image shape: {color_img.shape}")
        # color_img is RGB here
        save_image_cv(color_img, "dummy_color_saved.png") # Will be converted to BGR by save_image_cv

    # Clean up dummy files
    if os.path.exists(dummy_img_name): os.remove(dummy_img_name)
    if os.path.exists("dummy_grayscale_resized.png"): os.remove("dummy_grayscale_resized.png")
    if os.path.exists("dummy_color_image.png"): os.remove("dummy_color_image.png")
    if os.path.exists("dummy_color_saved.png"): os.remove("dummy_color_saved.png")
