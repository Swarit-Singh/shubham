import numpy as np
from PIL import Image
import os
import json
import logging
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2 # For some image operations if needed

# --- Logging Setup ---
def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Directory and File Handling ---
def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")

def load_config(config_path="config.json"):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from configuration file: {config_path}")
        return {}

# --- Image Loading and Saving (using PIL for Streamlit compatibility) ---
def load_image_pil(image_path, mode="L"): # L for grayscale, RGB for color
    try:
        img = Image.open(image_path)
        if mode:
            img = img.convert(mode)
        return np.array(img)
    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None

def save_image(image_array, save_path):
    """Saves a NumPy array as an image using PIL."""
    try:
        # Ensure array is in uint8 format for typical image saving
        if image_array.dtype != np.uint8:
            # Attempt to convert safely, e.g. clip and cast if float
            if np.issubdtype(image_array.dtype, np.floating):
                image_array = np.clip(image_array, 0, 255)
            image_array = image_array.astype(np.uint8)
        
        img = Image.fromarray(image_array)
        img.save(save_path)
        logging.info(f"Image saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving image to {save_path}: {e}")

# --- Watermark Bits Handling ---
def load_watermark_bits(file_path):
    """Loads watermark bits from .npy, .bin (raw bytes), or .txt (0s and 1s) files."""
    try:
        if file_path.endswith(".npy"):
            bits_array = np.load(file_path)
            return "".join(map(str, bits_array.astype(int)))
        elif file_path.endswith(".bin"):
            with open(file_path, 'rb') as f:
                byte_data = f.read()
            return "".join(format(byte, '08b') for byte in byte_data)
        elif file_path.endswith(".txt"):
            with open(file_path, 'r') as f:
                return f.read().strip()
        else:
            logging.warning(f"Unsupported file type for watermark bits: {file_path}")
            return None
    except Exception as e:
        logging.error(f"Error loading watermark bits from {file_path}: {e}")
        return None

def save_watermark_bits(bits_str, file_path, mode='txt'):
    """Saves watermark bits string to a file."""
    try:
        if mode == 'txt':
            with open(file_path, 'w') as f:
                f.write(bits_str)
        elif mode == 'npy':
            bits_array = np.array([int(b) for b in bits_str], dtype=np.uint8)
            np.save(file_path, bits_array)
        elif mode == 'bin':
            # Convert bit string to bytes
            byte_array = bytearray()
            for i in range(0, len(bits_str), 8):
                byte = bits_str[i:i+8]
                if len(byte) < 8: # Pad if not a full byte (might happen)
                    byte = byte.ljust(8, '0')
                byte_array.append(int(byte, 2))
            with open(file_path, 'wb') as f:
                f.write(byte_array)
        else:
            logging.warning(f"Unsupported save mode for watermark bits: {mode}")
            return
        logging.info(f"Watermark bits saved to {file_path} (mode: {mode})")
    except Exception as e:
        logging.error(f"Error saving watermark bits to {file_path}: {e}")


# --- Performance Metrics ---
def get_psnr(original_img, modified_img, max_pixel_val=255):
    """Calculates Peak Signal-to-Noise Ratio (PSNR) between two images."""
    if original_img.shape != modified_img.shape:
        logging.error("Images must have the same dimensions for PSNR.")
        # Attempt to resize modified_img to original_img's shape if drastically different
        # This is a fallback, ideally shapes should match.
        try:
            modified_img = cv2.resize(modified_img, (original_img.shape[1], original_img.shape[0]))
        except Exception as e:
            logging.error(f"Could not resize image for PSNR: {e}")
            return 0.0
            
    original_img = original_img.astype(np.float64)
    modified_img = modified_img.astype(np.float64)
    mse = np.mean((original_img - modified_img) ** 2)
    if mse == 0:
        return float('inf') # Should indicate perfect similarity
    return 20 * np.log10(max_pixel_val / np.sqrt(mse))

def get_ssim(original_img, modified_img, data_range=255):
    """Calculates Structural Similarity Index (SSIM) between two images."""
    if original_img.shape != modified_img.shape:
        logging.error("Images must have the same dimensions for SSIM.")
        try:
            modified_img = cv2.resize(modified_img, (original_img.shape[1], original_img.shape[0]))
        except Exception as e:
            logging.error(f"Could not resize image for SSIM: {e}")
            return 0.0

    # Ensure images are of compatible type for skimage.metrics.ssim
    original_img_sk = original_img.astype(np.uint8 if data_range==255 else original_img.dtype)
    modified_img_sk = modified_img.astype(np.uint8 if data_range==255 else modified_img.dtype)
    
    # For grayscale, ssim doesn't need channel_axis. If color, it might.
    # Assuming grayscale (H, W)
    return ssim(original_img_sk, modified_img_sk, data_range=data_range)


# --- Capacity Estimation Helpers (Simplified) ---
def get_embedding_capacity_pee(image_array, T):
    """Estimates embedding capacity for PEE based on prediction errors."""
    if T == 0: return 0 # Avoid division by zero or issues with T=0
    capacity = 0
    img = image_array.astype(np.int32)
    rows, cols = img.shape
    for r in range(rows):
        for c in range(1, cols):
            prediction = img[r, c-1]
            error = img[r, c] - prediction
            if -T <= error < T: # Expandable errors
                capacity += 1
    return capacity

def get_embedding_capacity_hs(image_array):
    """Estimates embedding capacity for Histogram Shifting."""
    hist, _ = np.histogram(image_array.flatten(), bins=np.arange(257))
    peak_val_count = np.max(hist) # Count of pixels at the peak
    return int(peak_val_count) # Each pixel at peak can store 1 bit

# --- Exception Handling Decorator ---
def handle_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Exception in {func.__name__}: {e}", exc_info=True)
            # Depending on context, you might re-raise, return None, or a default value
            return None 
    return wrapper

# Example usage of a decorated function:
# @handle_exceptions
# def risky_operation(data):
#     return data / 0 
# result = risky_operation(10) # Will log error and return None

if __name__ == '__main__':
    setup_logging()
    logging.info("Utils module test run.")
    
    # Test capacity estimation
    dummy_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    cap_pee = get_embedding_capacity_pee(dummy_image, T=1)
    cap_hs = get_embedding_capacity_hs(dummy_image)
    logging.info(f"Dummy Image PEE Capacity (T=1): {cap_pee}")
    logging.info(f"Dummy Image HS Capacity: {cap_hs}")

    # Test PSNR/SSIM
    dummy_image2 = dummy_image.copy()
    dummy_image2[0,0] += 10 # Introduce small change
    psnr_val = get_psnr(dummy_image, dummy_image2)
    ssim_val = get_ssim(dummy_image, dummy_image2)
    logging.info(f"PSNR (dummy vs modified): {psnr_val:.2f} dB")
    logging.info(f"SSIM (dummy vs modified): {ssim_val:.4f}")
    
    psnr_inf = get_psnr(dummy_image, dummy_image)
    logging.info(f"PSNR (dummy vs self): {psnr_inf}") # Should be inf
