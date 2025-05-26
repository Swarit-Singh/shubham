import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
import logging
from features import compute_prediction_errors, extract_global_features # Assuming these are in features.py
from utils import setup_logging

# setup_logging() # Or let app.py handle main logging setup

# --- Model Paths ---
# These are for a "classic" ML model (e.g., predicting PEE 'T' or capacity from statistical features)
# Distinct from the CNN+XGBoost model paths.
CLASSIC_RF_CAPACITY_MODEL_PATH = "classic_rf_capacity_predictor.pkl"
CLASSIC_RF_THRESHOLD_MODEL_PATH = "classic_rf_threshold_predictor.pkl"


def train_classic_ml_model(image_paths_list, target_values_list, model_type='capacity', save_path=None):
    """
    Trains a classic RandomForestRegressor model (e.g., for capacity or threshold prediction)
    based on global image features extracted from prediction errors.

    Args:
        image_paths_list (list of str): List of paths to training images.
        target_values_list (list of float): Corresponding target values (e.g., capacities, optimal T's).
        model_type (str): 'capacity' or 'threshold'. Determines default save path if not provided.
        save_path (str, optional): Path to save the trained model.
    Returns:
        sklearn.ensemble.RandomForestRegressor: The trained model, or None on error.
    """
    if len(image_paths_list) != len(target_values_list):
        logging.error("Number of images and target values must match.")
        return None
    if not image_paths_list:
        logging.error("Training data (image paths) is empty.")
        return None

    all_features = []
    valid_targets = []

    logging.info(f"Starting feature extraction for {len(image_paths_list)} images...")
    for i, img_path in enumerate(image_paths_list):
        try:
            # This assumes images are loaded as grayscale numpy arrays
            from PIL import Image # Using PIL for simple loading here
            image_arr = np.array(Image.open(img_path).convert('L'))
            
            pred_errors = compute_prediction_errors(image_arr)
            if pred_errors.size > 0:
                # Ensure features are returned in a consistent order (e.g., by sorting keys or using a fixed list)
                # For simplicity, assuming extract_global_features returns a dict, convert to a fixed-order array
                # This is crucial for scikit-learn models.
                feature_dict = extract_global_features(pred_errors)
                # Example: Define a fixed order of feature keys
                feature_keys = sorted(feature_dict.keys()) # Sort alphabetically for consistency
                feature_vector = [feature_dict[key] for key in feature_keys]
                
                all_features.append(feature_vector)
                valid_targets.append(target_values_list[i])
            else:
                logging.warning(f"Skipping image {img_path} due to empty prediction errors.")
        except Exception as e:
            logging.error(f"Error processing image {img_path} for feature extraction: {e}")
            continue # Skip this image

    if not all_features:
        logging.error("No features could be extracted from the provided images. Model training aborted.")
        return None

    X = np.array(all_features)
    y = np.array(valid_targets)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info(f"Training RandomForestRegressor for {model_type} prediction...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all cores
    rf_model.fit(X_train, y_train)

    # Evaluate model (optional, but good practice)
    y_pred_test = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    logging.info(f"Model training complete. Test MSE: {mse:.4f}")

    # Save the model
    if save_path is None:
        save_path = CLASSIC_RF_CAPACITY_MODEL_PATH if model_type == 'capacity' else CLASSIC_RF_THRESHOLD_MODEL_PATH
    
    try:
        joblib.dump(rf_model, save_path)
        logging.info(f"Trained '{model_type}' model saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving model to {save_path}: {e}")
        return rf_model # Return model even if saving failed

    return rf_model


def predict_with_classic_ml_model(image_array_or_path, model_path_or_obj):
    """
    Predicts using a loaded classic ML model.
    Args:
        image_array_or_path (np.ndarray or str): Input image array or path to image.
        model_path_or_obj (str or model object): Path to the saved model or the model object itself.
    Returns:
        float: Predicted value (e.g. capacity, threshold), or None on error.
    """
    # Load image if path is given
    if isinstance(image_array_or_path, str):
        try:
            from PIL import Image
            image_arr = np.array(Image.open(image_array_or_path).convert('L'))
        except Exception as e:
            logging.error(f"Failed to load image for prediction: {e}")
            return None
    elif isinstance(image_array_or_path, np.ndarray):
        image_arr = image_array_or_path
    else:
        logging.error("Invalid input: must be image array or path.")
        return None

    # Extract features
    pred_errors = compute_prediction_errors(image_arr)
    if pred_errors.size == 0:
        logging.warning("Cannot make prediction: empty prediction errors for input image.")
        return None
        
    feature_dict = extract_global_features(pred_errors)
    # Ensure feature order matches training (IMPORTANT!)
    feature_keys = sorted(feature_dict.keys()) 
    feature_vector = np.array([feature_dict[key] for key in feature_keys]).reshape(1, -1) # Reshape for single sample

    # Load model if path is given
    if isinstance(model_path_or_obj, str):
        if not os.path.exists(model_path_or_obj):
            logging.error(f"Model file not found: {model_path_or_obj}")
            return None
        try:
            model = joblib.load(model_path_or_obj)
        except Exception as e:
            logging.error(f"Error loading model from {model_path_or_obj}: {e}")
            return None
    else: # Assume it's already a model object
        model = model_path_or_obj
    
    if model is None:
        logging.error("Model is not loaded or provided.")
        return None

    try:
        prediction = model.predict(feature_vector)
        return prediction[0] # Prediction is usually an array, take the first element
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None


# Example usage for standalone testing
if __name__ == "__main__":
    setup_logging(level=logging.INFO)
    
    # Create dummy training data for demonstration
    dummy_image_dir = "dummy_classic_ml_train_data"
    if not os.path.exists(dummy_image_dir):
        os.makedirs(dummy_image_dir)
    
    num_dummy_images = 10
    dummy_img_paths = []
    dummy_capacities = [] # Example target: predict capacity
    dummy_thresholds = [] # Example target: predict optimal T

    logging.info(f"Creating {num_dummy_images} dummy images for classic ML training demo...")
    for i in range(num_dummy_images):
        path = os.path.join(dummy_image_dir, f"dummy_train_{i}.png")
        # Create images with varying characteristics to make features diverse
        img_content = np.random.randint(0, 50 + i * 20, (64, 64), dtype=np.uint8)
        Image.fromarray(img_content).save(path)
        dummy_img_paths.append(path)
        # Dummy targets - in reality, these would be derived from experiments or known values
        dummy_capacities.append(np.sum(img_content < 100) * 0.1 + np.random.rand() * 50) # Simplistic capacity
        dummy_thresholds.append(1 + (i % 4)) # Example optimal T could be 1, 2, 3, 4

    # --- Train a capacity prediction model ---
    logging.info("\n--- Training Classic ML Model for Capacity Prediction ---")
    trained_capacity_model = train_classic_ml_model(
        dummy_img_paths, 
        dummy_capacities, 
        model_type='capacity',
        save_path="dummy_capacity_model.pkl"
    )

    if trained_capacity_model:
        logging.info("Capacity model trained. Now testing prediction...")
        # Test prediction on the first dummy image
        predicted_cap = predict_with_classic_ml_model(dummy_img_paths[0], trained_capacity_model)
        if predicted_cap is not None:
            logging.info(f"Predicted capacity for {dummy_img_paths[0]}: {predicted_cap:.2f} (Actual dummy target: {dummy_capacities[0]:.2f})")
        if os.path.exists("dummy_capacity_model.pkl"):
             os.remove("dummy_capacity_model.pkl")


    # --- Train a threshold prediction model ---
    logging.info("\n--- Training Classic ML Model for Threshold Prediction ---")
    trained_threshold_model = train_classic_ml_model(
        dummy_img_paths,
        dummy_thresholds,
        model_type='threshold',
        save_path="dummy_threshold_model.pkl"
    )
    if trained_threshold_model:
        logging.info("Threshold model trained. Now testing prediction...")
        predicted_t = predict_with_classic_ml_model(dummy_img_paths[1], "dummy_threshold_model.pkl") # Test loading from path
        if predicted_t is not None:
            logging.info(f"Predicted threshold for {dummy_img_paths[1]}: {predicted_t:.2f} (Actual dummy target: {dummy_thresholds[1]})")
        if os.path.exists("dummy_threshold_model.pkl"):
            os.remove("dummy_threshold_model.pkl")

    # Clean up dummy image directory
    import shutil
    if os.path.exists(dummy_image_dir):
        shutil.rmtree(dummy_image_dir)
        logging.info(f"Cleaned up dummy directory: {dummy_image_dir}")
