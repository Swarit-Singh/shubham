import numpy as np
from PIL import Image
import argparse
import logging
import os

# Assuming your core operations are in watermark_operations and pipeline
from watermark_operations import extract_pred_error, extract_hist_shift, extract_ml_assisted, extract_ml_cnn_xgboost
# utils.py for loading, saving, metrics
from utils import load_image_pil, save_watermark_bits, get_psnr, get_ssim, setup_logging

# This utility might not use watermark_pipeline directly if it's only for extraction
# and assumes parameters are known or provided.

def extract_watermark_from_image(watermarked_image_path, original_image_path, output_payload_path,
                                 method, payload_length, params_dict=None):
    """
    Extracts a watermark from a watermarked image given the method and parameters.

    Args:
        watermarked_image_path (str): Path to the watermarked image file.
        original_image_path (str): Path to the original (cover) image file (for comparison/recovery metrics).
        output_payload_path (str): Path to save the extracted binary payload (e.g., payload.txt).
        method (str): The watermarking method used (e.g., 'prediction_error', 'histogram_shift').
        payload_length (int): The expected length of the payload in bits.
        params_dict (dict, optional): Dictionary of parameters used during embedding
                                      (e.g., {'T': 1} for PEE, {'p': peak, 'z': zero, 'direction': 'right'} for HS).
    Returns:
        bool: True if extraction was attempted, False on critical error before attempt.
    """
    setup_logging(level=logging.INFO) # Setup logging for the utility

    if not os.path.exists(watermarked_image_path):
        logging.error(f"Watermarked image not found: {watermarked_image_path}")
        return False
    if not os.path.exists(original_image_path):
        logging.warning(f"Original image not found: {original_image_path}. Metrics against original will not be calculated.")
        original_img_arr = None
    else:
        original_img_arr = load_image_pil(original_image_path, mode="L")
        if original_img_arr is None:
            logging.warning(f"Failed to load original image {original_image_path}.")


    watermarked_img_arr = load_image_pil(watermarked_image_path, mode="L")
    if watermarked_img_arr is None:
        logging.error(f"Failed to load watermarked image {watermarked_image_path}")
        return False

    if params_dict is None:
        params_dict = {} # Initialize if None

    extracted_bits = ""
    recovered_image_arr = None

    logging.info(f"Attempting to extract watermark using method: {method} with payload length: {payload_length}")
    logging.info(f"Parameters provided for extraction: {params_dict}")

    try:
        if method == 'prediction_error':
            T_val = params_dict.get('T', 1) # Default T=1 if not in params
            extracted_bits, recovered_image_arr = extract_pred_error(watermarked_img_arr, payload_length, T_val)
        elif method == 'histogram_shift':
            p_val = params_dict.get('p')
            z_val = params_dict.get('z')
            direction_val = params_dict.get('direction')
            if p_val is None or z_val is None or direction_val is None:
                logging.error("Histogram shifting parameters (p, z, direction) are missing.")
                return False
            extracted_bits, recovered_image_arr = extract_hist_shift(watermarked_img_arr, payload_length, p_val, z_val, direction_val)
        elif method == 'ml_assisted': # Assuming classic ML also uses PEE-like structure
            T_val = params_dict.get('T', 1)
            extracted_bits, recovered_image_arr = extract_ml_assisted(watermarked_img_arr, payload_length, T_val)
        elif method == 'ml_cnn_xgboost': # CNN+XGBoost also uses PEE-like structure with a predicted T
            T_val = params_dict.get('T', 1)
            extracted_bits, recovered_image_arr = extract_ml_cnn_xgboost(watermarked_img_arr, payload_length, T_val)
        else:
            logging.error(f"Unsupported extraction method: {method}")
            return False

        logging.info(f"Extraction complete. Extracted {len(extracted_bits)} bits.")
        if extracted_bits:
            logging.info(f"First 64 extracted bits: {extracted_bits[:64]}...")
            save_watermark_bits(extracted_bits, output_payload_path, mode='txt')
        else:
            logging.warning("No bits were extracted.")

        if recovered_image_arr is not None and original_img_arr is not None:
            psnr_rec = get_psnr(original_img_arr, recovered_image_arr)
            ssim_rec = get_ssim(original_img_arr, recovered_image_arr)
            logging.info(f"Recovery PSNR (Original vs Recovered): {psnr_rec:.2f} dB")
            logging.info(f"Recovery SSIM (Original vs Recovered): {ssim_rec:.4f}")
            if np.array_equal(original_img_arr, recovered_image_arr):
                logging.info("Image perfectly recovered.")
            else:
                logging.warning("Image not perfectly recovered.")
            
            # Optionally save the recovered image
            rec_img_path = os.path.join(os.path.dirname(output_payload_path), "recovered_image_standalone.png")
            Image.fromarray(recovered_image_arr).save(rec_img_path)
            logging.info(f"Recovered image saved to {rec_img_path}")

        return True

    except Exception as e:
        logging.error(f"An error occurred during watermark extraction: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Watermark Extraction Utility.")
    parser.add_argument("watermarked_image", help="Path to the watermarked image file.")
    parser.add_argument("original_image", help="Path to the original (cover) image file for metrics.")
    parser.add_argument("output_payload", help="Path to save the extracted payload (e.g., payload.txt).")
    parser.add_argument("--method", required=True, choices=['prediction_error', 'histogram_shift', 'ml_assisted', 'ml_cnn_xgboost'],
                        help="Watermarking method used for embedding.")
    parser.add_argument("--length", required=True, type=int, help="Expected length of the payload in bits.")
    
    # Parameters for PEE / ML-based PEE
    parser.add_argument("--T", type=int, help="Threshold T for PEE-based methods.")
    
    # Parameters for HS
    parser.add_argument("--p", type=int, help="Peak value 'p' for Histogram Shifting.")
    parser.add_argument("--z", type=int, help="Zero/min value 'z' for Histogram Shifting.")
    parser.add_argument("--direction", choices=['left', 'right', 'none'], help="Shift direction for Histogram Shifting.")

    args = parser.parse_args()

    # Construct params_dict from command-line arguments
    extraction_params = {}
    if args.T is not None: extraction_params['T'] = args.T
    if args.p is not None: extraction_params['p'] = args.p
    if args.z is not None: extraction_params['z'] = args.z
    if args.direction is not None: extraction_params['direction'] = args.direction

    extract_watermark_from_image(
        args.watermarked_image,
        args.original_image,
        args.output_payload,
        args.method,
        args.length,
        params_dict=extraction_params
    )
    # Example command:
    # python extract_watermark.py watermarked.png original.png extracted_payload.txt --method prediction_error --length 128 --T 1
    # python extract_watermark.py watermarked_hs.png original.png extracted_hs_payload.txt --method histogram_shift --length 100 --p 150 --z 180 --direction right
