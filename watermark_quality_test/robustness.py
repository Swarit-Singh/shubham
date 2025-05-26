import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from PIL import Image
import io

# Adjust this import based on your project structure if watermark_pipeline is not in parent dir
# If robustness.py is in watermark_quality_test/ and watermark_pipeline.py is in root:
from watermark_pipeline import watermark_pipeline



def test_single_watermarking_scenario(cover_img_np: np.ndarray,
                                      payload_bits: list,
                                      method: str,
                                      **kwargs): # For method-specific params like T for PEE
    """
    Tests a single watermarking scenario for basic metrics and robustness.
    Returns a dictionary of results.
    """
    results = {
        "method_tested": method,
        "payload_length": len(payload_bits),
        "params_passed_to_pipeline": kwargs
    }

    # --- 1. Clean Embedding & Extraction ---
    try:
        wm_clean, rec_clean, bits_ext_clean, embed_p = watermark_pipeline(
            cover_img_np, payload_bits, method, **kwargs
        )
        results["embedding_params"] = embed_p
        results["psnr_clean_wm_vs_input"] = float('inf') if np.array_equal(cover_img_np, wm_clean) else \
                                        psnr_metric(cover_img_np, wm_clean, data_range=255)
        
        min_dim_clean = min(cover_img_np.shape[:2])
        win_s_clean = 7 if min_dim_clean >= 7 else max(3, (min_dim_clean // 2 * 2 + 1))
        results["ssim_clean_wm_vs_input"] = ssim_metric(cover_img_np, wm_clean, data_range=255, channel_axis=None, win_size=win_s_clean)
        
        results["perfect_image_recovery_clean"] = np.array_equal(cover_img_np, rec_clean)
        
        correct_clean = 0
        if len(payload_bits) > 0 and len(bits_ext_clean) >= len(payload_bits):
            correct_clean = np.sum(np.array(payload_bits) == bits_ext_clean[:len(payload_bits)])
            results["acc_clean"] = correct_clean / len(payload_bits)
        elif len(payload_bits) == 0:
             results["acc_clean"] = 1.0 # No bits to embed, 100% accuracy
        else:
            results["acc_clean"] = 0.0 # Extracted fewer bits than payload

    except Exception as e:
        results["error_clean_processing"] = str(e)
        results["acc_clean"] = 0.0
        results["psnr_clean_wm_vs_input"] = 0.0
        results["ssim_clean_wm_vs_input"] = 0.0
        results["perfect_image_recovery_clean"] = False
        wm_clean = cover_img_np # Fallback for further tests if clean embedding fails

    # --- 2. JPEG Robustness (Quality 75) ---
    try:
        pil_wm_clean = Image.fromarray(wm_clean.astype(np.uint8))
        buffer = io.BytesIO()
        pil_wm_clean.save(buffer, format="JPEG", quality=75)
        jpeg_reloaded_pil = Image.open(buffer).convert("L")
        jpeg_reloaded_np = np.array(jpeg_reloaded_pil)

        _, _, bits_ext_jpeg, _ = watermark_pipeline(
            jpeg_reloaded_np, payload_bits, method, **kwargs # Use original payload for extraction attempt
        )
        correct_jpeg = 0
        if len(payload_bits) > 0 and len(bits_ext_jpeg) >= len(payload_bits):
            correct_jpeg = np.sum(np.array(payload_bits) == bits_ext_jpeg[:len(payload_bits)])
            results["acc_jpeg"] = correct_jpeg / len(payload_bits)
        elif len(payload_bits) == 0:
            results["acc_jpeg"] = 1.0
        else:
             results["acc_jpeg"] = 0.0
    except Exception as e:
        results["error_jpeg_processing"] = str(e)
        results["acc_jpeg"] = 0.0

    # --- 3. Gaussian Noise Robustness (sigma=5) ---
    try:
        noise = np.random.normal(0, 5, wm_clean.shape)
        noisy_wm_np = np.clip(wm_clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        _, _, bits_ext_noise, _ = watermark_pipeline(
            noisy_wm_np, payload_bits, method, **kwargs
        )
        correct_noise = 0
        if len(payload_bits) > 0 and len(bits_ext_noise) >= len(payload_bits):
            correct_noise = np.sum(np.array(payload_bits) == bits_ext_noise[:len(payload_bits)])
            results["acc_noise"] = correct_noise / len(payload_bits)
        elif len(payload_bits) == 0:
            results["acc_noise"] = 1.0
        else:
            results["acc_noise"] = 0.0
    except Exception as e:
        results["error_noise_processing"] = str(e)
        results["acc_noise"] = 0.0
        
    return results
