from watermark_operations import (
    embed_pred_error, extract_pred_error,
    embed_hist_shift, extract_hist_shift,
    embed_ml_cnn_xgboost, extract_ml_cnn_xgboost,
)

def watermark_pipeline(image_array, payload_bits_str, method, **kwargs):
    """
    Orchestrates the watermarking process: embedding and then extraction for verification.
    Args:
        image_array (np.ndarray): The grayscale input image.
        payload_bits_str (str): The binary payload string.
        method (str): The watermarking method identifier.
        **kwargs: Method-specific parameters (e.g., T for PEE).
    Returns:
        tuple: (watermarked_image, recovered_image, extracted_bits, embedding_params)
    """
    watermarked_image = None
    recovered_image = None
    extracted_bits = ""
    embedding_params = {}

    if not payload_bits_str: # Handle empty payload gracefully
        return image_array.copy(), image_array.copy(), "", {}

    if method == "prediction_error":
        # Only use manual/slider T, no ML
        T_val = kwargs.get("T", 1)
        watermarked_image, embedding_params = embed_pred_error(image_array, payload_bits_str, T=T_val)
        len_to_extract = embedding_params.get('embedded_count', len(payload_bits_str))
        extracted_bits, recovered_image = extract_pred_error(watermarked_image, len_to_extract, T=T_val)
    
    elif method == "histogram_shift":
        # Only use classic HS logic, no ML
        watermarked_image, embedding_params = embed_hist_shift(image_array, payload_bits_str)
        len_to_extract = embedding_params.get('embedded_count', len(payload_bits_str))
        p = embedding_params.get('p')
        z = embedding_params.get('z')
        direction = embedding_params.get('direction')
        if p is not None and z is not None and direction is not None:
            extracted_bits, recovered_image = extract_hist_shift(watermarked_image, len_to_extract, p, z, direction)
        else:
            recovered_image = watermarked_image
            extracted_bits = "" 
            
    elif method == "ml_cnn_xgboost":
        # Only ML method uses T predicted by ML model (passed in kwargs)
        T_val = kwargs.get("T", 1)
        watermarked_image, embedding_params = embed_ml_cnn_xgboost(image_array, payload_bits_str, T=T_val)
        len_to_extract = embedding_params.get('embedded_count', len(payload_bits_str))
        extracted_bits, recovered_image = extract_ml_cnn_xgboost(watermarked_image, len_to_extract, T=T_val)
        
    else:
        raise ValueError(f"Unknown watermarking method: {method}")

    return watermarked_image, recovered_image, extracted_bits, embedding_params
