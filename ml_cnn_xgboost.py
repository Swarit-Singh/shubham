import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0 # EfficientNetB0 is relatively small
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import Model # Not strictly needed if using base_model directly
import xgboost as xgb
import joblib
import os

# Define model paths (consider making these configurable or constants)
XGB_MODEL_PATH = "xgb_threshold_predictor.pkl"
# CNN model isn't saved here as we load pre-trained EfficientNet each time.
# If you fine-tune the CNN, you'd save and load it.

# Global variable for CNN model to avoid reloading if not necessary
# This is a simple optimization; for more complex apps, consider a class or context manager.
_cnn_model_global = None

def load_cnn_model(force_reload=False):
    """Loads the CNN feature extractor model (EfficientNetB0)."""
    global _cnn_model_global
    if _cnn_model_global is None or force_reload:
        # Using include_top=False to get features, pooling='avg' to get a flat vector
        _cnn_model_global = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
        # print("CNN Model (EfficientNetB0) loaded.") # For debugging
    return _cnn_model_global

def extract_cnn_features(image_array_grayscale, cnn_model):
    """
    Extracts features from a grayscale image array using the pre-loaded CNN model.
    Args:
        image_array_grayscale (np.ndarray): Grayscale image (H, W).
        cnn_model: Loaded Keras CNN model.
    Returns:
        np.ndarray: Flattened feature vector.
    """
    if image_array_grayscale.ndim == 2: # Grayscale
        # Convert grayscale to 3-channel by replicating the channel
        image_rgb_array = np.stack([image_array_grayscale]*3, axis=-1)
    elif image_array_grayscale.ndim == 3 and image_array_grayscale.shape[2] == 1: # Grayscale (H, W, 1)
        image_rgb_array = np.concatenate([image_array_grayscale]*3, axis=-1)
    elif image_array_grayscale.ndim == 3 and image_array_grayscale.shape[2] == 3: # Already RGB
        image_rgb_array = image_array_grayscale
    else:
        raise ValueError("Input image must be grayscale (H,W) or (H,W,1), or RGB (H,W,3)")

    # Resize to CNN input size (e.g., 224x224 for EfficientNetB0)
    pil_img = Image.fromarray(image_rgb_array.astype(np.uint8)).resize((224, 224))
    
    # Convert to Keras array format and preprocess
    img_keras_array = img_to_array(pil_img)
    img_keras_array = np.expand_dims(img_keras_array, axis=0) # Add batch dimension
    img_keras_array = preprocess_input(img_keras_array) # Preprocessing specific to EfficientNet

    features = cnn_model.predict(img_keras_array, verbose=0) # verbose=0 to suppress prediction progress bar
    return features.flatten()

def train_xgboost_model(list_of_feature_vectors, list_of_thresholds):
    """
    Trains an XGBoost regressor model and saves it.
    Args:
        list_of_feature_vectors (list of np.ndarray): CNN features for training images.
        list_of_thresholds (list of float): Corresponding target thresholds.
    """
    X_train = np.array(list_of_feature_vectors)
    y_train = np.array(list_of_thresholds)

    # Initialize and train XGBoost regressor
    # Parameters can be tuned for better performance
    xgb_regressor = xgb.XGBRegressor(
        objective="reg:squarederror", # For regression
        n_estimators=100,             # Number of trees
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    xgb_regressor.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(xgb_regressor, XGB_MODEL_PATH)
    # print(f"XGBoost model trained and saved to {XGB_MODEL_PATH}") # For debugging

def load_xgboost_model():
    """Loads the pre-trained XGBoost model."""
    if not os.path.exists(XGB_MODEL_PATH):
        # print(f"Warning: XGBoost model file not found at {XGB_MODEL_PATH}. Returning None.") # For debugging
        return None # Or raise an error, or return a default untrained model
    return joblib.load(XGB_MODEL_PATH)

def predict_threshold_xgboost(single_feature_vector, xgb_model):
    """
    Predicts a threshold using the loaded XGBoost model.
    Args:
        single_feature_vector (np.ndarray): CNN features for a single image.
        xgb_model: Loaded XGBoost model.
    Returns:
        float: Predicted threshold.
    """
    if xgb_model is None:
        # print("Warning: XGBoost model is not loaded. Returning default threshold 1.") # For debugging
        return 1.0 # Default fallback if model isn't available
        
    # XGBoost expects a 2D array for prediction [n_samples, n_features]
    prediction = xgb_model.predict(np.array([single_feature_vector]))
    return float(prediction[0])

# Example of how to prepare data and train (can be run as a script for initial model training)
if __name__ == "__main__":
    # This part is for standalone testing or initial model creation.
    # In the Streamlit app, training is triggered by a button.
    print("Testing ml_cnn_xgboost.py functions...")

    # Create dummy training data folder if it doesn't exist
    dummy_data_folder = "ml_training_data_dummy"
    if not os.path.exists(dummy_data_folder):
        os.makedirs(dummy_data_folder)
        # Create a few dummy images and label files
        for i in range(3):
            dummy_img = Image.fromarray(np.random.randint(0, 256, (100, 100), dtype=np.uint8))
            dummy_img.save(os.path.join(dummy_data_folder, f"dummy_img_{i}.png"))
            with open(os.path.join(dummy_data_folder, f"dummy_img_{i}.txt"), "w") as f_label:
                f_label.write(str(np.random.randint(1, 5))) # Dummy threshold 1-4
        print(f"Created dummy training data in ./{dummy_data_folder}/")


    # --- Test Training ---
    print("\n--- Testing Training ---")
    if os.path.isdir(dummy_data_folder) and len(os.listdir(dummy_data_folder)) > 1:
        train_images = []
        train_labels = []
        for fname in os.listdir(dummy_data_folder):
            if fname.lower().endswith(".png"):
                img_p = os.path.join(dummy_data_folder, fname)
                lbl_p = os.path.join(dummy_data_folder, fname.rsplit('.',1)[0] + ".txt")
                if os.path.exists(lbl_p):
                    train_images.append(np.array(Image.open(img_p).convert("L")))
                    with open(lbl_p, "r") as f:
                        train_labels.append(float(f.read().strip()))
        
        if train_images:
            cnn = load_cnn_model()
            print("Extracting features for dummy images...")
            dummy_features = [extract_cnn_features(img, cnn) for img in train_images]
            print(f"Extracted {len(dummy_features)} feature sets. First feature vector shape: {dummy_features[0].shape}")
            
            print("Training XGBoost model with dummy data...")
            train_xgboost_model(dummy_features, train_labels) # This will save xgb_threshold_predictor.pkl
            print("XGBoost model training complete (dummy).")
        else:
            print("No dummy images/labels found for training test.")
    else:
        print(f"Dummy data folder '{dummy_data_folder}' not found or empty. Skipping training test.")


    # --- Test Prediction ---
    print("\n--- Testing Prediction ---")
    if os.path.exists(XGB_MODEL_PATH):
        xgb_loaded = load_xgboost_model()
        if xgb_loaded and train_images: # Ensure dummy_features is available
            # Use the first dummy image's features for prediction test
            test_feature_vec = extract_cnn_features(train_images[0], load_cnn_model(force_reload=True)) # Ensure cnn model is fresh if needed
            pred_t = predict_threshold_xgboost(test_feature_vec, xgb_loaded)
            print(f"Predicted threshold for the first dummy image: {pred_t:.2f} (Actual was: {train_labels[0]})")
        else:
            print("Could not load XGBoost model or no features to test prediction.")
    else:
        print(f"{XGB_MODEL_PATH} not found. Run training first or ensure the file exists.")
    
    # Clean up dummy data
    # import shutil
    # if os.path.exists(dummy_data_folder):
    #     shutil.rmtree(dummy_data_folder)
    #     print(f"Cleaned up dummy folder: {dummy_data_folder}")
