import numpy as np
from scipy import stats
import logging

# --- Feature Extraction from Prediction Errors (for classic ML) ---

def compute_prediction_errors(image_array, predictor='left'):
    """
    Computes prediction errors for an image.
    Args:
        image_array (np.ndarray): Grayscale input image.
        predictor (str): 'left', 'top', 'median_edge' (simple predictors).
    Returns:
        np.ndarray: Array of prediction errors.
    """
    errors = []
    img = image_array.astype(np.int32) # Work with integers for error calculation
    rows, cols = img.shape

    for r in range(rows):
        for c in range(cols):
            if predictor == 'left':
                if c == 0: continue # Skip first column
                pred_val = img[r, c-1]
            elif predictor == 'top':
                if r == 0: continue # Skip first row
                pred_val = img[r-1, c]
            # Add more complex predictors if needed
            # elif predictor == 'median_edge': # Example GAP / MED predictor
            #     if r == 0 or c == 0: continue
            #     neighbors = [img[r, c-1], img[r-1, c], img[r-1, c-1]]
            #     # Simplified: use average of left and top
            #     pred_val = (img[r,c-1] + img[r-1,c]) // 2
            else: # Default to left if predictor unknown
                if c == 0: continue
                pred_val = img[r, c-1]
            
            error = img[r, c] - pred_val
            errors.append(error)
            
    return np.array(errors)

def extract_global_features(prediction_errors_array):
    """
    Extracts global statistical features from a 1D array of prediction errors.
    Args:
        prediction_errors_array (np.ndarray): 1D array of prediction errors.
    Returns:
        dict: Dictionary of features. Returns empty dict if input is empty.
    """
    if prediction_errors_array.size == 0:
        logging.warning("Prediction error array is empty. Cannot extract features.")
        # Return a dictionary of NaNs or zeros for fixed-length feature vector
        return {
            'mean_error': np.nan, 'std_error': np.nan, 'abs_mean_error': np.nan,
            'skewness': np.nan, 'kurtosis': np.nan,
            'p_error_0': np.nan, 'p_error_pm1': np.nan, 'p_error_pm2': np.nan,
            'entropy': np.nan
        }

    features = {}
    features['mean_error'] = np.mean(prediction_errors_array)
    features['std_error'] = np.std(prediction_errors_array)
    features['abs_mean_error'] = np.mean(np.abs(prediction_errors_array))
    
    # Skewness and Kurtosis
    features['skewness'] = stats.skew(prediction_errors_array)
    features['kurtosis'] = stats.kurtosis(prediction_errors_array) # Fisher's definition (normal is 0)

    # Probability of specific errors (useful for PEE capacity)
    features['p_error_0'] = np.sum(prediction_errors_array == 0) / len(prediction_errors_array)
    features['p_error_pm1'] = np.sum(np.abs(prediction_errors_array) == 1) / len(prediction_errors_array)
    features['p_error_pm2'] = np.sum(np.abs(prediction_errors_array) == 2) / len(prediction_errors_array)

    # Entropy of prediction errors (approximated)
    # Discretize errors for entropy calculation or use histogram
    hist_errors, _ = np.histogram(prediction_errors_array, bins=50, density=True) # Bins can be optimized
    hist_errors = hist_errors[hist_errors > 0] # Remove zero probabilities
    features['entropy'] = -np.sum(hist_errors * np.log2(hist_errors) * (np.diff(_)[0] if len(_) > 1 else 1)) # Shannon entropy

    return features

# Alias for consistency if used elsewhere with this name
extract_features = extract_global_features 

# Example Usage
if __name__ == '__main__':
    from utils import setup_logging
    setup_logging(level=logging.DEBUG)

    dummy_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    
    # Compute errors
    pred_errors = compute_prediction_errors(dummy_image, predictor='left')
    logging.debug(f"Computed {len(pred_errors)} prediction errors. Example errors: {pred_errors[:10]}")

    # Extract features
    if pred_errors.size > 0:
        image_features = extract_global_features(pred_errors)
        logging.info("Extracted Features:")
        for key, val in image_features.items():
            logging.info(f"  {key}: {val:.4f}")
    else:
        logging.warning("No prediction errors computed, skipping feature extraction display.")

    # Test with empty errors
    empty_errors = np.array([])
    empty_features = extract_global_features(empty_errors)
    logging.info(f"Features from empty errors: {empty_features}")
